# dashboard.py
import streamlit as st
import requests
import pandas as pd

# Título do Dashboard
st.title("Delegacia 5.0 - Mini-Dashboard de Análise Preditiva")
st.write("Use os campos abaixo para simular uma ocorrência e obter a previsão do tipo de crime.")

# URL da sua API FastAPI (que deve estar rodando em outro terminal)
API_URL = "http://127.0.0.1:8000/predict"

# Criando as colunas para o layout
col1, col2 = st.columns(2)

with col1:
    st.header("Detalhes da Ocorrência")
    bairro = st.text_input("Bairro", "Boa Viagem")
    arma = st.selectbox("Arma Utilizada", ["Arma de Fogo", "Arma Branca", "Nenhuma"])
    hora = st.slider("Hora do Dia", 0, 23, 20)
    qtd_vitimas = st.number_input("Quantidade de Vítimas", min_value=0, max_value=10, value=1)

with col2:
    st.header("Localização e Suspeitos")
    lat = st.number_input("Latitude", value=-8.1299, format="%.4f")
    lon = st.number_input("Longitude", value=-34.9035, format="%.4f")
    sexo_suspeito = st.radio("Sexo do Suspeito", ["Masculino", "Feminino", "Desconhecido"])
    qtd_suspeitos = st.number_input("Quantidade de Suspeitos", min_value=0, max_value=10, value=2)

# Botão para enviar os dados para a API
if st.button("Analisar e Prever Tipo de Crime"):
    # Estrutura do JSON que a API espera
    payload = {
      "bairro": bairro,
      "descricao_modus_operandi": "Simulado via Dashboard", # Valor fixo para simplificar
      "arma_utilizada": arma,
      "sexo_suspeito": sexo_suspeito,
      "orgao_responsavel": "Delegacia de Roubos e Furtos", # Valor fixo
      "status_investigacao": "Em Andamento", # Valor fixo
      "quantidade_vitimas": qtd_vitimas,
      "quantidade_suspeitos": qtd_suspeitos,
      "idade_suspeito": 25, # Valor fixo
      "latitude": lat,
      "longitude": lon,
      "ano": 2024, # Valor fixo
      "mes": 10,   # Valor fixo
      "dia_semana": 5, # Valor fixo
      "hora": hora
    }

    try:
        # Fazendo a chamada POST para a sua API
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()
            st.success(f"**Tipo de Crime Previsto:**")
            st.metric(label="Classificação do Modelo", value=prediction['tipo_crime_predito'])
        else:
            st.error(f"Erro ao chamar a API: {response.status_code}")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar à API. Verifique se ela está rodando em http://127.0.0.1:8000")