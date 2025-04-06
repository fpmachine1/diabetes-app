
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Estilo de fundo
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Verificador de Diabetes</h1>", unsafe_allow_html=True)
st.markdown("### 🤖 Preencha os dados abaixo para descobrir a probabilidade de diabetes.")

# Carregar o modelo
with open('modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Inputs organizados
col1, col2, col3 = st.columns(3)
with col1:
    gravidez = st.number_input("Gravidez", 0, 20)
    espessura = st.number_input("Espessura da pele", 0, 100)
with col2:
    glicose = st.number_input("Glicose", 0, 300)
    insulina = st.number_input("Insulina", 0, 900)
with col3:
    pressao = st.number_input("Pressão Sanguínea", 0, 200)
    imc = st.number_input("IMC", 0.0, 70.0, step=0.1)

col4, col5 = st.columns(2)
with col4:
    pedigree = st.number_input("Função pedigree do diabetes", 0.000, 2.500, step=0.001, format="%.3f")
with col5:
    idade = st.number_input("Idade", 0, 120)

st.markdown("---")

# Botão de verificação
if st.button("🔎 Verificar"):
    dados = np.array([[gravidez, glicose, pressao, espessura, insulina, imc, pedigree, idade]])
    resultado = modelo.predict(dados)

    st.markdown("## Resultado:")
    if resultado[0] == 1:
        st.error("🚨 Alta probabilidade de diabetes!")
    else:
        st.success("✅ Baixa probabilidade de diabetes.")
