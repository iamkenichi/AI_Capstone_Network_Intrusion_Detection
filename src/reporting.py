"""Generate auditable tables, matplotlib figures and explanations from executed models."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from src.data_loader import ROOT, load_partition, predictors, fingerprints

FIG = ROOT/'figures'
REP = ROOT/'reports'


def save(name):
    plt.tight_layout()
    plt.savefig(FIG/f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close()


def table(frame):
    return frame.to_markdown(index=False, floatfmt='.4f')


def eda(train):
    plt.rcParams.update({'figure.figsize':(10,5.5), 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.titleweight':'bold', 'axes.titlesize':15,
        'font.size':11, 'axes.prop_cycle':plt.cycler(color=['#127D92','#E47C45','#5765A1','#9F416D'])})
    raw = load_partition('training')
    test = load_partition('testing')
    stats = []
    for name, df in [('Published training',raw),('Published test',test),('Development train',train)]:
        stats.append(dict(partition=name, rows=len(df), columns=len(df.columns),
            missing_cells=int(df.isna().sum().sum()), exact_duplicates=int(df.duplicated().sum()),
            predictor_duplicates=int(fingerprints(df).duplicated().sum()),
            benign=int((df.label==0).sum()),attack=int(df.label.sum())))
    pd.DataFrame(stats).to_csv(REP/'dataset_summary.csv',index=False)
    raw.dtypes.astype(str).rename('dtype').to_csv(REP/'feature_types.csv')
    raw.isna().sum().rename('missing_count').to_csv(REP/'missing_values.csv')
    raw.attack_cat.value_counts().rename('count').to_csv(REP/'attack_distribution.csv')
    desc = train.select_dtypes(include=np.number).describe(percentiles=[.01,.25,.5,.75,.99]).T
    desc.to_csv(REP/'numeric_summary.csv')
    n = predictors(train).select_dtypes(include=np.number)
    q1,q3 = n.quantile(.25),n.quantile(.75)
    out = ((n < q1-1.5*(q3-q1)) | (n > q3+1.5*(q3-q1))).mean().sort_values(ascending=False)
    out.rename('iqr_outlier_fraction').to_csv(REP/'outlier_analysis.csv')
    (n<0).sum().rename('negative_count').to_csv(REP/'negative_values.csv')
    train.label.map({0:'Benign',1:'Attack'}).value_counts().plot.bar(rot=0)
    plt.title('Class balance | development training partition'); plt.ylabel('Flows'); save('class_distribution')
    train.attack_cat.value_counts().sort_values().plot.barh()
    plt.title('Attack categories | development training partition'); plt.xlabel('Flows'); save('attack_categories')
    for column in ['proto','service','state']:
        grouped = train.groupby(column).label.agg(['size','mean']).sort_values('size',ascending=False).head(12)
        grouped.to_csv(REP/f'eda_{column}.csv')
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        grouped['size'].plot.bar(ax=axes[0]); axes[0].set_title(f'{column}: flow counts'); axes[0].set_ylabel('Flows')
        grouped['mean'].plot.bar(ax=axes[1]); axes[1].set_title(f'{column}: observed attack fraction'); axes[1].set_ylim(0,1)
        save(f'{column}_attack_rate')
    selected = ['dur','sbytes','dbytes','spkts','rate','sttl','dttl','tcprtt']
    fig,axes = plt.subplots(2,4,figsize=(15,7))
    for ax,col in zip(axes.flat,selected):
        for label,color in [(0,'#127D92'),(1,'#E47C45')]:
            values=np.log1p(train.loc[train.label==label,col])
            ax.hist(values,bins=40,alpha=.5,density=True,color=color,label=['Benign','Attack'][label])
        ax.set_title(col); ax.set_xlabel('log(1 + value)'); ax.set_ylabel('Density')
    axes.flat[0].legend(); save('feature_distributions')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,col in zip(axes,['sbytes','dur','spkts']):
        ax.boxplot([np.log1p(train.loc[train.label==i,col]) for i in [0,1]],tick_labels=['Benign','Attack'],showfliers=False)
        ax.set_title(col); ax.set_ylabel('log(1 + value)')
    save('feature_boxplots')
    corr=n.corr(method='spearman'); corr.to_csv(REP/'spearman_correlations.csv')
    fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=7)
    ax.set_yticks(range(len(corr)),corr.columns,fontsize=7)
    fig.colorbar(im,ax=ax,label='Spearman correlation'); ax.set_title('Flow-feature relationships | training only'); save('correlations')
    association=n.apply(lambda x:x.corr(train.label,method='spearman')).dropna().sort_values(key=abs,ascending=False)
    association.rename('spearman_with_target').to_csv(REP/'target_association.csv')
    association.head(12).sort_values().plot.barh(); plt.xlabel('Spearman correlation with attack label')
    plt.title('Target association | training only'); save('target_association')
    behavior=train.groupby('attack_cat')[['dur','sbytes','dbytes','spkts','dpkts','rate','sttl','tcprtt']].median()
    behavior.to_csv(REP/'attack_category_behavior.csv')
    logged=np.log1p(behavior)
    normalized=(logged-logged.mean())/logged.std().replace(0,1)
    fig,ax=plt.subplots(figsize=(10,6))
    im=ax.imshow(normalized,cmap='RdBu_r',aspect='auto',vmin=-2,vmax=2)
    ax.set_xticks(range(len(behavior.columns)),behavior.columns,rotation=45)
    ax.set_yticks(range(len(behavior)),behavior.index)
    fig.colorbar(im,ax=ax,label='Standardized log(1 + category median)')
    ax.set_title('Attack-category behavior | training only'); save('attack_category_behavior')
    sample=train.sample(min(2500,len(train)),random_state=42)
    plt.scatter(np.log1p(sample.sbytes),np.log1p(sample.dbytes),c=sample.label,cmap='coolwarm',s=8,alpha=.4)
    plt.xlabel('log(1 + source bytes)'); plt.ylabel('log(1 + destination bytes)')
    plt.title('Traffic directionality | sampled training flows'); save('byte_relationship')
    feature_dictionary(raw)
    return pd.DataFrame(stats)


def feature_dictionary(raw):
    original=pd.read_csv(ROOT/'data/raw/NUSW-NB15_features.csv',encoding='cp1252')
    source={str(r['Name']).lower().replace(' ',''):str(r['Description']).strip() for _,r in original.iterrows()}
    aliases={'smean':'smeansz','dmean':'dmeansz','response_body_len':'res_bdy_len','sinpkt':'sintpkt','dinpkt':'dintpkt'}
    manual={'id':'Partition-local record identifier; no predictive role.',
            'rate':'Flow packet rate supplied by the dataset; preserve the provided unit convention.'}
    rows=[]
    for col in raw.columns:
        categorical=col in ['proto','service','state']
        excluded=col in ['id','label','attack_cat']
        role={'id':'excluded identifier','label':'binary target','attack_cat':'audit only'}.get(col,'predictor')
        category=('annotation' if excluded else 'categorical' if categorical else
            'connection context' if col.startswith('ct_') else 'TCP' if col in ['swin','dwin','stcpb','dtcpb','tcprtt','synack','ackdat'] else 'flow statistic')
        meaning=('Never use as model input.' if excluded else
            'May encode collection topology or operating-system defaults; audit transferability.' if 'ttl' in col or col in ['stcpb','dtcpb'] else
            'Context counts require matching, causal flow aggregation during deployment.' if col.startswith('ct_') else
            'Describes traffic behavior; association alone does not establish malicious intent.')
        rows.append(dict(feature_name=col,data_type=str(raw[col].dtype),category=category,
            description=manual.get(col,source.get(aliases.get(col,col),'Definition requires source verification.')),
            role=role,preprocessing=('Excluded from X' if excluded else 'Train-fitted mode, rare-category one-hot encoding' if categorical else
                'Finite values; train-fitted median and standard scaling'),possible_security_meaning=meaning))
    engineered={
        'total_bytes':('sbytes + dbytes','Bidirectional volume'),
        'total_packets':('spkts + dpkts','Bidirectional packet count'),
        'bytes_per_packet':('total_bytes / total_packets','Average packet payload/overhead proxy'),
        'source_byte_share':('sbytes / total_bytes','Directional byte imbalance'),
        'source_packet_share':('spkts / total_packets','Directional packet imbalance'),
        'bytes_per_second':('total_bytes / dur','Volume relative to duration')}
    for name,(formula,meaning) in engineered.items():
        rows.append(dict(feature_name=name,data_type='float64',category='engineered',description=formula,
            role='predictor',preprocessing='Undefined ratios -> missing -> training median; standard scaling',possible_security_meaning=meaning))
    pd.DataFrame(rows).to_csv(REP/'data_dictionary.csv',index=False)


def evaluation_figures(test,metadata):
    probabilities=np.load(ROOT/'data/processed/test_probabilities.npz')
    comparison=pd.read_csv(REP/'model_comparison.csv')
    fig,ax=plt.subplots()
    for name in probabilities.files: RocCurveDisplay.from_predictions(test.label,probabilities[name],name=name,ax=ax)
    ax.set_title('ROC curves | primary test partition'); save('roc_curves')
    fig,ax=plt.subplots()
    for name in probabilities.files: PrecisionRecallDisplay.from_predictions(test.label,probabilities[name],name=name,ax=ax)
    ax.axhline(test.label.mean(),linestyle='--',color='gray',label='Attack prevalence'); ax.legend()
    ax.set_title('Precision–recall curves | primary test partition'); save('pr_curves')
    for _,row in comparison.iterrows():
        ConfusionMatrixDisplay.from_predictions(test.label,probabilities[row.model]>=row.threshold,
            display_labels=['Benign','Attack'],cmap='Blues',values_format=',d')
        plt.title(f'{row.model} | threshold {row.threshold:.2f}'); save('confusion_'+row.model.lower().replace(' ','_'))
    comparison.set_index('model')[['precision','recall','f1','fpr']].plot.bar(rot=0,figsize=(11,5))
    plt.legend(loc='upper left',bbox_to_anchor=(1.01,1))
    plt.ylim(0,1.05); plt.ylabel('Metric'); plt.title('Detection and false-alert trade-offs | primary test'); save('model_comparison')
    thresholds=pd.read_csv(REP/f'threshold_{metadata["model"].lower().replace(" ","_")}.csv')
    thresholds.plot(x='threshold',y=['recall','precision','fpr','fnr'])
    plt.axvline(metadata['threshold'],linestyle='--',color='black'); plt.title('Operating threshold | validation only')
    plt.ylabel('Metric'); save('threshold_tradeoff')


def explain(test, models, metadata):
    import shap
    val=pd.read_csv(REP/'validation_comparison.csv')
    best=val[val.model.isin(['Random Forest','XGBoost'])].sort_values('pr_auc',ascending=False).iloc[0]
    pipeline=models[best.model]
    X=predictors(test)
    p=pipeline.predict_proba(X)[:,1]
    predicted=p>=best.threshold
    cases={'true_positive':(test.label.to_numpy()==1)&predicted,
           'true_negative':(test.label.to_numpy()==0)&~predicted,
           'false_positive':(test.label.to_numpy()==0)&predicted,
           'false_negative':(test.label.to_numpy()==1)&~predicted}
    rng=np.random.default_rng(42)
    selected=list(rng.choice(len(test),size=min(160,len(test)),replace=False))
    local={name:int(np.flatnonzero(mask)[0]) for name,mask in cases.items() if mask.any()}
    indices=list(dict.fromkeys(selected+list(local.values())))
    transformed=pipeline[:-1].transform(X.iloc[indices])
    names=pipeline.named_steps['preprocess'].get_feature_names_out()
    explainer=shap.TreeExplainer(pipeline.named_steps['model'])
    explanation=explainer(transformed)
    if explanation.values.ndim==3:
        explanation=explanation[:,:,1]
    explanation.feature_names=list(names)
    reconstructed=explanation.base_values+explanation.values.sum(axis=1)
    if best.model=='XGBoost':
        reconstructed=1/(1+np.exp(-reconstructed))
    np.testing.assert_allclose(reconstructed,p[indices],atol=2e-5,rtol=2e-5)
    shap.plots.beeswarm(explanation[:len(selected)],max_display=15,show=False); save('shap_beeswarm')
    shap.plots.bar(explanation[:len(selected)],max_display=15,show=False); save('shap_importance')
    importance=pd.DataFrame({'feature':names,'mean_abs_shap':np.abs(explanation.values[:len(selected)]).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    importance.to_csv(REP/'shap_importance.csv',index=False)
    local_records=[]
    for name,pos in local.items():
        k=indices.index(pos)
        shap.plots.waterfall(explanation[k],max_display=12,show=False); save('shap_'+name)
        values=explanation.values[k]
        highest=np.argsort(np.abs(values))[-6:][::-1]
        local_records.append(dict(case=name,row_id=int(test.iloc[pos].id),label=int(test.iloc[pos].label),
            attack_cat=test.iloc[pos].attack_cat,probability=float(p[pos]),
            strongest=[dict(feature=str(names[i]),shap=float(values[i])) for i in highest]))
    result=dict(model=best.model,threshold=float(best.threshold),random_sample_size=len(selected),
        missing_cases=sorted(set(cases)-set(local)),local=local_records,
        units='Log odds' if best.model=='XGBoost' else 'Probability')
    (REP/'shap_details.json').write_text(json.dumps(result,indent=2))
    return result


def generate(train,validation,test,models,metadata):
    stats=eda(train)
    evaluation_figures(test,metadata)
    print('Generating SHAP explanations',flush=True)
    details=explain(test,models,metadata)
    from src.write_documents import write_all
    write_all(stats,metadata,details)


if __name__=='__main__':
    import joblib
    metadata=json.loads((ROOT/'models/metadata.json').read_text())
    models={name:joblib.load(ROOT/f'models/{name.lower().replace(" ","_")}.joblib')
            for name in ['Logistic Regression','Random Forest','XGBoost']}
    frames=[pd.read_csv(ROOT/f'data/processed/{name}.csv') for name in ['train','validation','test']]
    generate(*frames,models,metadata)
