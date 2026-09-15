"""Local demonstration for already extracted UNSW-NB15 flow features."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from src.data_loader import ROOT, predictors
from src.predict import load_model, predict

st.set_page_config(page_title='Network Intrusion Research',page_icon='🛡️',layout='wide')
st.title('Network intrusion detection')
st.caption('UNSW-NB15 · Research prototype · Analyst decision support')
st.warning('This model is an educational/research prototype and should not be used as a standalone production intrusion-detection system.')


@st.cache_resource
def resources():
    return load_model()


try:
    model, metadata=resources()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()

st.write(f"Model: **{metadata['model']}** · Operating threshold: **{metadata['threshold']:.2f}**")
st.write('Upload extracted flow features or edit a provided example. Raw packet captures are not supported.')
upload=st.file_uploader('Network flows (CSV)',type='csv')
try:
    if upload is not None:
        frame=pd.read_csv(upload)
    else:
        examples=pd.read_csv(ROOT/'app/example_flows.csv')
        index=st.selectbox('Example flow',range(len(examples)),format_func=lambda i:f'Example {i+1}')
        frame=examples.iloc[[index]].copy().reset_index(drop=True)
    frame=st.data_editor(frame,num_rows='fixed',use_container_width=True)
    if st.button('Analyze flows',type='primary'):
        result=predict(frame,model,metadata)
        st.dataframe(result,use_container_width=True)
        st.download_button('Download predictions',result.to_csv(index=False),'predictions.csv','text/csv')
        st.info('Attack means the flow crosses the research alert threshold and merits analyst review. Benign means it is below that threshold; it is not proof of safety. Probabilities are uncalibrated model scores, not incident severity or expected financial loss.')
        with st.expander('What influences the model?'):
            st.image(str(ROOT/'figures/shap_importance.png'))
            st.caption('Global SHAP importance from sampled test flows; this is not a local explanation of the uploaded flow.')
except (ValueError,KeyError,pd.errors.ParserError) as error:
    st.error(f'Input could not be analyzed: {error}')
