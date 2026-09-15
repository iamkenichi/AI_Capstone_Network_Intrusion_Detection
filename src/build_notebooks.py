"""Six executable notebooks provide a narrated, checked view of the saved experiment."""
import nbformat as nbf
from src.data_loader import ROOT, predictors
import pandas as pd


def build():
    intro="""from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown
from src.data_loader import load_partition, predictors, fingerprints
def report(name):
    display(Markdown((ROOT / 'reports' / name).read_text(encoding='utf-8')))
def figure(name):
    display(Image(filename=str(ROOT / 'figures' / name)))
"""
    notebooks=[
        ('01_problem_data_understanding','Problem and data understanding',[
            ('m','Define binary flow classification, its decision context and research targets. Data labels are kept out of predictors.'),
            ('c',"report('problem_statement.md')"),
            ('c',"train_raw=load_partition('training')\ntest_raw=load_partition('testing')\nassert len(train_raw)==175341 and len(test_raw)==82332\ndisplay(pd.read_csv(ROOT/'reports/dataset_summary.csv'))\ndisplay(train_raw.dtypes.to_frame('type'))"),
            ('c',"display(train_raw.isna().sum().to_frame('missing'))\ndisplay(train_raw.label.value_counts().to_frame('count'))\ndisplay(train_raw.attack_cat.value_counts().to_frame('count'))\ndisplay(pd.read_csv(ROOT/'reports/data_dictionary.csv'))"),
            ('m','Interpretation: raw partition totals differ from the cleaned modeling populations. Predictor duplicates ignore the partition-local ID. Provenance normalizes a filename reversal in the mirror; official checksum equivalence remains unverified.'),
            ('c',"report('dataset_documentation.md')")]),
        ('02_eda','Exploratory data analysis',[
            ('c',"train=pd.read_csv(ROOT/'data/processed/train.csv')\ndisplay(train.describe(include='all'))"),
            ('c',"figure('class_distribution.png')\nfigure('attack_categories.png')"),
            ('m','Interpretation: class and category counts reveal representation. Overall binary performance cannot establish rare-attack coverage. Weighting is compared during tuning; no synthetic oversampling is applied.'),
            ('c',"figure('attack_category_behavior.png')\ndisplay(pd.read_csv(ROOT/'reports/attack_category_behavior.csv'))"),
            ('m','Interpretation: category medians compare traffic volume, duration, TTL and TCP behavior. Color normalizes the logged medians within each feature for visual comparison; it does not indicate significance or causal attack signatures.'),
            ('c',"for name in ['proto_attack_rate.png','service_attack_rate.png','state_attack_rate.png']:\n    figure(name)"),
            ('m','Interpretation: category attack fractions are conditioned on this collection. Compare rates with support before attributing risk to any protocol or service. They are not stable security rules.'),
            ('c',"figure('feature_distributions.png')\nfigure('feature_boxplots.png')\nfigure('byte_relationship.png')"),
            ('m','Interpretation: log(1+x) reveals skew and overlap in numeric features. Extremes are retained because large volumes or unusual timing can describe attack traffic. Boxplots hide individual outlier markers for readability; the outlier table retains their counts.'),
            ('c',"figure('correlations.png')\nfigure('target_association.png')\ndisplay(pd.read_csv(ROOT/'reports/outlier_analysis.csv').head(15))"),
            ('m','Interpretation: correlation suggests redundancy and possible environment shortcuts. It neither proves leakage nor justifies feature selection using final-test outcomes.'),
            ('c',"report('EDA_Feature_Engineering_Report.md')")]),
        ('03_preprocessing_feature_engineering','Preprocessing and feature engineering',[
            ('c',"from src.features import NetworkFeatures\ntrain=pd.read_csv(ROOT/'data/processed/train.csv')\nvalidation=pd.read_csv(ROOT/'data/processed/validation.csv')\ntest=pd.read_csv(ROOT/'data/processed/test.csv')\nassert not set(fingerprints(train)) & set(fingerprints(validation))\nassert not set(fingerprints(pd.concat([train,validation]))) & set(fingerprints(test))\nX=predictors(train)\nassert not {'label','attack_cat','id'} & set(X.columns)\ndisplay(NetworkFeatures().fit_transform(X.head()))"),
            ('c',"import joblib\nmodel=joblib.load(ROOT/'models/final_model.joblib')\ntransformed=model[:-1].transform(X.head(100))\nassert np.isfinite(transformed).all()\nprint('Original predictors:',X.shape[1], 'Transformed columns:',transformed.shape[1])\ndisplay(pd.read_csv(ROOT/'reports/feature_ablation.csv'))"),
            ('m','Interpretation: median/mode/scaling/encoding parameters come from training. Ratios with zero denominators remain undefined until imputation. The validation ablation keeps hyperparameters fixed and does not reopen test-based model selection.')]),
        ('04_model_training','Model training and selection',[
            ('m','The expensive training experiment was executed with `python -m src.train`. This notebook inspects its reproducible configuration and saved results; rerun that command to retrain. The primary selection metric is average precision.'),
            ('c',"metadata=json.loads((ROOT/'models/metadata.json').read_text())\ndisplay(metadata)\nfor name in ['logistic_regression','random_forest','xgboost']:\n    cv=pd.read_csv(ROOT/f'reports/tuning_{name}.csv')\n    display(cv[['params','mean_train_score','mean_test_score','std_test_score','rank_test_score']])"),
            ('c',"validation=pd.read_csv(ROOT/'reports/validation_comparison.csv')\ndisplay(validation)\nassert validation.sort_values('pr_auc',ascending=False).iloc[0]['model']==metadata['model']"),
            ('m','Interpretation: four candidates and three folds per family provide a bounded comparison. Training versus CV scores help inspect overfitting, but a small search is not proof of global optimality. Validation may itself be optimistic because it supports model and threshold selection.')]),
        ('05_model_evaluation','Evaluation and operating threshold',[
            ('c',"comparison=pd.read_csv(ROOT/'reports/model_comparison.csv')\ndisplay(comparison)\ndisplay(pd.read_csv(ROOT/'reports/model_comparison_threshold_050.csv'))\nreport('Model_Evaluation_Report.md')"),
            ('c',"figure('roc_curves.png')\nfigure('pr_curves.png')\nfigure('model_comparison.png')"),
            ('m','Interpretation: discrimination scores do not specify alert workload at the operating threshold. Compare precision and FPR alongside attack recall. The winner was fixed by validation, even if another family has a higher test F1.'),
            ('c',"for name in ['logistic_regression','random_forest','xgboost']:\n    figure(f'confusion_{name}.png')\nfigure('threshold_tradeoff.png')"),
            ('m','Interpretation: threshold curves use validation only. Test false-alert rates can exceed validation rates under distribution shift. Unmet operating constraints require disclosure and further local validation, not test-based threshold retuning.'),
            ('c',"from src.evaluate import metrics\nimport joblib\nfrom src.predict import predict\ntest=pd.read_csv(ROOT/'data/processed/test.csv')\nmodel=joblib.load(ROOT/'models/final_model.joblib')\nmeta=json.loads((ROOT/'models/metadata.json').read_text())\nresult=predict(test.iloc[:100],model,meta)\nassert len(result)==100\ndisplay(result.head())\nprint('Published test secondary:',json.loads((ROOT/'reports/published_test_secondary.json').read_text()))")]),
        ('06_explainability_bias_audit','Explainability and operational audit',[
            ('c',"figure('shap_beeswarm.png')\nfigure('shap_importance.png')"),
            ('m','Interpretation: the beeswarm shows signed associations in the fitted tree model; the bar chart shows magnitude. TTL/context reliance may encode collection conditions. SHAP does not establish causality or demographic fairness.'),
            ('c',"for outcome in ['true_positive','true_negative','false_positive','false_negative']:\n    figure(f'shap_{outcome}.png')\nreport('Explainability_Report.md')"),
            ('m','Interpretation: local cases are real observations selected by outcome, not fabricated demonstrations. They illustrate why scores moved but do not estimate how common each explanation is.'),
            ('c',"audit=pd.read_csv(ROOT/'reports/subgroup_audit.csv')\ndisplay(audit[audit.group_type=='attack_cat'])\nassert audit.loc[(audit.group_type=='attack_cat') & (audit.group!='Normal'),'fpr'].isna().all()\nreport('Bias_Fairness_Analysis.md')"),
            ('m','Interpretation: group support and uncertainty matter. Attack-only groups have undefined FPR; no demographic protected attributes are available. Mitigations require new evaluation data and human governance.')])]
    for filename,title,blocks in notebooks:
        notebook=nbf.v4.new_notebook()
        notebook.cells=[nbf.v4.new_markdown_cell('# '+title),nbf.v4.new_code_cell(intro)]
        notebook.cells += [nbf.v4.new_markdown_cell(text) if kind=='m' else nbf.v4.new_code_cell(text) for kind,text in blocks]
        notebook.metadata={'kernelspec':{'display_name':'Python 3 (capstone)','language':'python','name':'python3'},
                           'language_info':{'name':'python','version':'3.13'}}
        nbf.write(notebook,ROOT/f'notebooks/{filename}.ipynb')
    train=pd.read_csv(ROOT/'data/processed/train.csv')
    examples=train.groupby('label',group_keys=False).head(3)
    predictors(examples).to_csv(ROOT/'app/example_flows.csv',index=False)


if __name__=='__main__':
    build()
