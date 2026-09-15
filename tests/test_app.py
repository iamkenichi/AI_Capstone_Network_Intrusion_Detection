from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_loads_and_predicts_example():
    root=Path(__file__).resolve().parents[1]
    if not (root/'models/final_model.joblib').exists():
        import pytest
        pytest.skip('Integration test requires the trained artifact.')
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    result=app.dataframe[-1].value
    assert 'attack_probability' in result.columns
    assert result.attack_probability.between(0,1).all()
