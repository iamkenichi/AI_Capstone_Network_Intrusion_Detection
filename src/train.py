"""One command for fixed-split training, validation selection and final evaluation."""
import argparse
import json
import time
from pathlib import Path
import platform
import importlib.metadata
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from src.data_loader import ROOT, load_partition, predictors, fingerprints, deduplicate
from src.preprocessing import build_pipeline
from src.evaluate import metrics, threshold_table, choose_threshold, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-rows', type=int, default=30000,
                        help='Stratified training subset for bounded tuning; refit uses ALL training rows.')
    parser.add_argument('--iterations', type=int, default=4)
    args = parser.parse_args()
    np.random.seed(42)
    for folder in ['models', 'reports', 'figures', 'data/processed']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = load_partition('training'), load_partition('testing')
    if (len(raw_train), len(raw_test)) != (175341, 82332):
        raise ValueError('Expected canonical 175341 training and 82332 test rows. Run scripts/download_data.py.')
    clean, conflicts = deduplicate(raw_train)
    train, validation = train_test_split(clean, test_size=0.2, stratify=clean.label, random_state=42)
    test_unique, test_conflicts = deduplicate(raw_test)
    overlap = fingerprints(test_unique).isin(set(fingerprints(clean)))
    test = test_unique.loc[~overlap].copy()
    split_info = dict(raw_train=len(raw_train), raw_test=len(raw_test), clean_development=len(clean),
        training=len(train), validation=len(validation), primary_test=len(test),
        removed_development=len(raw_train)-len(clean), conflicting_development_signatures=conflicts,
        removed_test_duplicates_or_conflicts=len(raw_test)-len(test_unique),
        conflicting_test_signatures=test_conflicts, test_overlap_removed=int(overlap.sum()), seed=42)
    (ROOT/'reports/split_manifest.json').write_text(json.dumps(split_info, indent=2))
    for name, frame in [('train',train), ('validation',validation), ('test',test)]:
        frame.to_csv(ROOT / f'data/processed/{name}.csv', index=False)
    if len(train) > args.search_rows:
        search, _ = train_test_split(train, train_size=args.search_rows, stratify=train.label, random_state=42)
    else:
        search = train
    X, y = predictors(train), train.label
    Xs, ys = predictors(search), search.label
    Xv, yv = predictors(validation), validation.label
    Xt, yt = predictors(test), test.label
    weight = float((ys == 0).sum()/(ys == 1).sum())
    candidates = {
        'Logistic Regression': (LogisticRegression(max_iter=2500, random_state=42),
            {'model__C':[0.01,0.1,1,10], 'model__class_weight':[None,'balanced']}),
        'Random Forest': (RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=4),
            {'model__max_depth':[12,20,None], 'model__min_samples_leaf':[1,3,8],
             'model__class_weight':[None,'balanced']}),
        'XGBoost': (XGBClassifier(n_estimators=220, tree_method='hist', random_state=42, n_jobs=4,
                                 eval_metric='logloss'),
            {'model__max_depth':[3,5,7], 'model__learning_rate':[0.05,0.1],
             'model__subsample':[0.8,1.0], 'model__scale_pos_weight':[1.0,weight]})}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    validation_rows, models, timing = [], {}, {}
    for name, (estimator, grid) in candidates.items():
        print(f'Tuning {name} on {len(search):,} training rows', flush=True)
        start = time.perf_counter()
        searcher = RandomizedSearchCV(build_pipeline(estimator), grid, n_iter=args.iterations,
            scoring='average_precision', cv=cv, random_state=42, n_jobs=1, refit=False,
            error_score='raise', return_train_score=True)
        with threadpool_limits(limits=4):
            searcher.fit(Xs, ys)
            fitted = build_pipeline(clone(estimator)).set_params(**searcher.best_params_).fit(X, y)
        seconds = time.perf_counter()-start
        key = name.lower().replace(' ', '_')
        pd.DataFrame(searcher.cv_results_).to_csv(ROOT/f'reports/tuning_{key}.csv', index=False)
        pv = fitted.predict_proba(Xv)[:,1]
        table = threshold_table(yv, pv)
        threshold, feasible = choose_threshold(table)
        table.to_csv(ROOT/f'reports/threshold_{key}.csv', index=False)
        validation_rows.append(dict(model=name, threshold=threshold, constraint_met=feasible,
            cv_pr_auc=float(searcher.best_score_), **metrics(yv,pv,threshold)))
        timing[name] = dict(training_seconds=seconds, best_params=searcher.best_params_)
        models[name] = fitted
        joblib.dump(fitted, ROOT/f'models/{key}.joblib', compress=3)
        print(f'{name}: validation AP={validation_rows[-1]["pr_auc"]:.4f}, threshold={threshold}', flush=True)
    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(ROOT/'reports/validation_comparison.csv', index=False)
    # Selection is finalized BEFORE accessing any final-test predictions.
    selected = validation_table.sort_values('pr_auc', ascending=False).iloc[0]
    best_name, best_threshold = str(selected.model), float(selected.threshold)
    metadata = dict(model=best_name, threshold=best_threshold, features=list(X.columns),
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=42, selection='Highest validation average precision; validation-only threshold',
        constraint_met=bool(selected.constraint_met), search_rows=len(search), iterations=args.iterations,
        cv_folds=3, timing=timing, python=platform.python_version(),
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','xgboost','shap']})
    (ROOT/'models/metadata.json').write_text(json.dumps(metadata, indent=2))
    joblib.dump(models[best_name], ROOT/'models/final_model.joblib', compress=3)
    # An untuned feature ablation uses the selected hyperparameters and validation only.
    baseline = build_pipeline(clone(models[best_name].named_steps['model']), engineered=False)
    with threadpool_limits(limits=4):
        baseline.fit(X,y)
    ablation = pd.DataFrame([dict(features='Original', **metrics(yv,baseline.predict_proba(Xv)[:,1])),
        dict(features='Engineered', **metrics(yv,models[best_name].predict_proba(Xv)[:,1]))])
    ablation.to_csv(ROOT/'reports/feature_ablation.csv', index=False)
    comparison, default_comparison, probabilities = [], [], {}
    for row in validation_rows:
        name = row['model']
        start = time.perf_counter()
        p = models[name].predict_proba(Xt)[:,1]
        elapsed = time.perf_counter()-start
        probabilities[name] = p
        comparison.append(dict(model=name, threshold=row['threshold'], **metrics(yt,p,row['threshold']),
            training_seconds=timing[name]['training_seconds'], inference_seconds=elapsed,
            inference_ms_per_row=elapsed/len(test)*1000))
        default_comparison.append(dict(model=name, **metrics(yt,p)))
    pd.DataFrame(comparison).to_csv(ROOT/'reports/model_comparison.csv', index=False)
    pd.DataFrame(default_comparison).to_csv(ROOT/'reports/model_comparison_threshold_050.csv', index=False)
    p = probabilities[best_name]
    predictions = test[['id','label','attack_cat','proto','service','state']].copy()
    predictions['probability'] = p
    predictions['prediction'] = (p >= best_threshold).astype(int)
    predictions.to_csv(ROOT/'reports/test_predictions.csv', index=False)
    audit(test,p,best_threshold).to_csv(ROOT/'reports/subgroup_audit.csv', index=False)
    # Secondary published-split result transparently includes overlaps/duplicates.
    secondary = metrics(raw_test.label,models[best_name].predict_proba(predictors(raw_test))[:,1],best_threshold)
    (ROOT/'reports/published_test_secondary.json').write_text(json.dumps(secondary,indent=2))
    np.savez_compressed(ROOT/'data/processed/test_probabilities.npz', **probabilities)
    from src.reporting import generate
    generate(train, validation, test, models, metadata)
    print(json.dumps({'selected_model':best_name, 'threshold':best_threshold,
                      'primary_test':next(r for r in comparison if r['model']==best_name)},indent=2), flush=True)


if __name__ == '__main__':
    main()
