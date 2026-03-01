import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Sistema de Diagnóstico - Obesidade", page_icon="🩺", layout="wide")
st.title("🩺 Sistema Preditivo e Analítico de Obesidade")
st.markdown("Bem-vindo ao painel de apoio à decisão médica. Navegue pelas abas abaixo.")

@st.cache_resource
def load_artifacts():
    modelo = joblib.load('modelo_obesidade.pkl')
    scaler = joblib.load('scaler_numerico.pkl')
    encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    return modelo, scaler, encoders, target_encoder

@st.cache_data
def load_data():
    return pd.read_csv('Obesity.csv')

modelo_rf, scaler, label_encoders, le_target = load_artifacts()
df = load_data()

aba1, aba2 = st.tabs(["📊 Dashboard Analítico", "🤖 Diagnóstico Preditivo"])

with aba1:
    st.header("Análise Exploratória dos Dados")
    st.write("Visão geral do perfil dos pacientes na base de dados histórica.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_target = px.histogram(df, x='Obesity', title="Distribuição dos Níveis de Peso/Obesidade", color='Obesity')
        st.plotly_chart(fig_target, use_container_width=True)
        
        fig_fam = px.histogram(df, x='Obesity', color='family_history', barmode='group', 
                               title="Impacto do Histórico Familiar na Obesidade")
        st.plotly_chart(fig_fam, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(df, x='Weight', y='Height', color='Gender', 
                                 title="Relação: Peso vs Altura por Gênero")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        fig_transp = px.box(df, x='MTRANS', y='Age', color='Obesity', 
                            title="Meio de Transporte e Idade vs Obesidade")
        st.plotly_chart(fig_transp, use_container_width=True)

with aba2:
    st.header("Simulador de Diagnóstico")
    st.write("Insira os dados do paciente abaixo para prever o risco de obesidade.")
    
    with st.form("form_paciente"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.subheader("Dados Biométricos")
            gender = st.selectbox("Gênero (Gender)", ['Female', 'Male'])
            age = st.number_input("Idade (Age)", min_value=14, max_value=100, value=25)
            height = st.number_input("Altura em metros (Height)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Peso em kg (Weight)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            family_history = st.selectbox("Histórico Familiar de Sobrepeso?", ['yes', 'no'])
            
        with col_b:
            st.subheader("Hábitos Alimentares")
            favc = st.selectbox("Consome alimentos calóricos frequentemente? (FAVC)", ['yes', 'no'])
            fcvc = st.slider("Frequência de vegetais nas refeições (FCVC)", 1, 3, 2)
            ncp = st.slider("Número de refeições principais (NCP)", 1, 4, 3)
            caec = st.selectbox("Come entre as refeições? (CAEC)", ['no', 'Sometimes', 'Frequently', 'Always'])
            ch2o = st.slider("Consumo diário de água (CH2O)", 1, 3, 2)
            
        with col_c:
            st.subheader("Estilo de Vida")
            smoke = st.selectbox("Fumante? (SMOKE)", ['yes', 'no'])
            scc = st.selectbox("Monitora calorias diárias? (SCC)", ['yes', 'no'])
            faf = st.slider("Frequência de atividade física semanal (FAF)", 0, 3, 1)
            tue = st.slider("Tempo diário em telas/dispositivos (TUE)", 0, 2, 1)
            calc = st.selectbox("Consumo de álcool (CALC)", ['no', 'Sometimes', 'Frequently', 'Always'])
            mtrans = st.selectbox("Meio de transporte (MTRANS)", ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])
            
        submit_button = st.form_submit_button(label="Gerar Diagnóstico Preditivo")

    if submit_button:
        dados_input = pd.DataFrame([[
            gender, age, height, weight, family_history, favc, fcvc, ncp, caec, 
            smoke, ch2o, scc, faf, tue, calc, mtrans
        ]], columns=[
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC', 'NCP', 'CAEC', 
            'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
        ])
        
        for col in label_encoders.keys():
            dados_input[col] = label_encoders[col].transform(dados_input[col])
            
        dados_input[['Age', 'Height', 'Weight']] = scaler.transform(dados_input[['Age', 'Height', 'Weight']])
        
        previsao_codificada = modelo_rf.predict(dados_input)
        
        diagnostico = le_target.inverse_transform(previsao_codificada)[0]
        
        st.markdown("---")
        st.subheader("Resultado do Modelo:")
        if "Normal" in diagnostico:
            st.success(f"### O diagnóstico previsto é: **{diagnostico}**")
        elif "Overweight" in diagnostico:
            st.warning(f"### O diagnóstico previsto é: **{diagnostico}**")
        else:
            st.error(f"### O diagnóstico previsto é: **{diagnostico}**")