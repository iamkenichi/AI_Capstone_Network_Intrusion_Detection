"""Independent artifact checks after training, notebook execution and pytest."""
from pathlib import Path
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluate import metrics
from src.data_loader import fingerprints


def main():
    required=['README.md','requirements.txt','requirements-lock.txt','environment.yml','.gitignore','LICENSE',
        'reports/problem_statement.md','reports/dataset_documentation.md','reports/data_dictionary.csv',
        'reports/EDA_Feature_Engineering_Report.md','reports/Model_Evaluation_Report.md',
        'reports/Bias_Fairness_Analysis.md','reports/Final_Project_Report.md','reports/Generative_AI_Usage.md',
        'presentations/technical_presentation_content.md','presentations/executive_presentation_content.md',
        'app/streamlit_app.py','models/final_model.joblib','models/metadata.json']
    assert all((ROOT/name).is_file() for name in required)
    for record in json.loads((ROOT/'reports/data_provenance.json').read_text()):
        assert hashlib.sha256((ROOT/'data/raw'/record['file']).read_bytes()).hexdigest()==record['sha256']
    frames={name:pd.read_csv(ROOT/f'data/processed/{name}.csv') for name in ['train','validation','test']}
    signatures={name:set(fingerprints(df)) for name,df in frames.items()}
    assert not signatures['train'] & signatures['validation']
    assert not (signatures['train']|signatures['validation']) & signatures['test']
    meta=json.loads((ROOT/'models/metadata.json').read_text())
    assert not set(meta['features']) & {'id','label','attack_cat'}
    pred=pd.read_csv(ROOT/'reports/test_predictions.csv')
    computed=metrics(pred.label,pred.probability,meta['threshold'])
    saved=pd.read_csv(ROOT/'reports/model_comparison.csv').set_index('model').loc[meta['model']]
    for key,value in computed.items():
        np.testing.assert_allclose(value,saved[key],rtol=1e-6,atol=1e-7)
    dictionary=pd.read_csv(ROOT/'reports/data_dictionary.csv')
    assert len(dictionary)==51 and dictionary.notna().all().all()
    assert not dictionary.description.str.contains('requires source verification').any()
    notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==6
    cells=0
    for path in notebooks:
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                cells+=1
                assert cell.execution_count is not None, path.name
                assert not any(output.output_type=='error' for output in cell.outputs),path.name
    for audience,count in [('technical',12),('executive',10)]:
        text=(ROOT/f'presentations/{audience}_presentation_content.md').read_text(encoding='utf-8')
        assert len(re.findall(r'^## Slide ',text,re.M))==count
        for label in ['### Main content','### Recommended visual','### Speaker notes']:
            assert text.count(label)==count
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',readme):
        if not link.startswith(('https://','http://','#')):
            assert (ROOT/link.split('#')[0]).exists(),link
    figures=list((ROOT/'figures').glob('*.png'))
    for path in figures:
        with Image.open(path) as im:
            im.verify()
    pytest=ET.parse(ROOT/'reports/pytest_results.xml').getroot()
    suites=list(pytest.iter('testsuite'))
    assert sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)==0
    result={'status':'passed','input_checksums':3,'original_predictors':len(meta['features']),
        'dictionary_rows':len(dictionary),'executed_notebooks':len(notebooks),'executed_code_cells':cells,
        'valid_figures':len(figures),'pytest_tests':sum(int(s.get('tests',0)) for s in suites),
        'pytest_skipped':sum(int(s.get('skipped',0)) for s in suites),
        'split_overlap_check':'passed','saved_metric_recalculation':'passed','README_local_links':'passed'}
    (ROOT/'reports/artifact_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
