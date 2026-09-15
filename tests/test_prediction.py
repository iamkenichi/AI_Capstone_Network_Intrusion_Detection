import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src.preprocessing import build_pipeline
from src.predict import predict
from src.evaluate import metrics, choose_threshold, threshold_table
from tests.test_preprocessing import flows


def test_prediction_round_trip_and_saved_threshold(tmp_path):
    X=flows(); model=build_pipeline(LogisticRegression()).fit(X,[0,0,0,1,1,1])
    path=tmp_path/'model.joblib'; joblib.dump(model,path)
    metadata={'features':list(X.columns),'threshold':0.0}
    result=predict(X,joblib.load(path),metadata)
    assert set(result.prediction)=={'Attack'}
    np.testing.assert_allclose(result.attack_probability,model.predict_proba(X)[:,1])
    with pytest.raises(ValueError,match='Missing required'):
        predict(X.drop(columns='sbytes'),model,metadata)
    with pytest.raises(ValueError,match='Missing required'):
        predict(X.drop(columns='proto'),model,metadata)
    invalid=X.copy(); invalid['sbytes']='-1'
    with pytest.raises(ValueError,match='nonnegative'):
        predict(invalid,model,metadata)


def test_undefined_subgroup_metrics_and_counts():
    result=metrics([1,1],[0.2,0.8])
    assert np.isnan(result['fpr']) and result['fnr']==0.5
    assert result['fn']==1 and result['tp']==1


def test_threshold_constraint_fallback_is_explicit():
    threshold, feasible=choose_threshold(threshold_table([0,0,1,1],[0.9,0.9,0.1,0.1]))
    assert not feasible and 0.01<=threshold<=0.99
