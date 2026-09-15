import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.data_loader import predictors, deduplicate
from src.features import NetworkFeatures
from src.preprocessing import build_pipeline


def flows():
    return pd.DataFrame({'sbytes':[0,10,30,300,500,800], 'dbytes':[0,20,10,2,1,0],
        'spkts':[0,2,3,5,6,8], 'dpkts':[0,2,1,1,1,0], 'dur':[0,1,2,3,4,5],
        'proto':['tcp','tcp','udp','udp','tcp','udp'], 'service':['-']*6, 'state':['FIN']*6})


def test_safe_features_and_input_preservation():
    frame=flows(); original=frame.copy(deep=True)
    out=NetworkFeatures().fit_transform(frame)
    assert np.isnan(out.loc[0,'bytes_per_second'])
    assert not np.isinf(out.select_dtypes(include=np.number)).any().any()
    assert out.loc[1,'source_byte_share']==1/3
    assert out.loc[1,'total_packets']==4
    pd.testing.assert_frame_equal(frame,original)


def test_unseen_categories_and_training_only_statistics():
    frame=flows(); model=build_pipeline(LogisticRegression()).fit(frame,[0,0,0,1,1,1])
    stats=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
    unseen=frame.iloc[[0]].copy(); unseen['proto']='unseen'; unseen['sbytes']=np.nan
    assert np.isfinite(model.predict_proba(unseen)).all()
    np.testing.assert_array_equal(stats,model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_)


def test_target_and_identifier_exclusion():
    frame=flows().assign(label=[0,0,0,1,1,1],attack_cat='label_proxy',id=range(6))
    assert not {'label','attack_cat','id'} & set(predictors(frame))


def test_duplicates_ignore_identifier_and_remove_conflicting_labels():
    frame=flows().iloc[[1,1,2]].copy().reset_index(drop=True)
    frame['id']=[1,2,3]; frame['label']=[0,1,0]; frame['attack_cat']=['Normal','Attack','Normal']
    clean, conflicts=deduplicate(frame)
    assert conflicts==1 and len(clean)==1 and clean.iloc[0].id==3
