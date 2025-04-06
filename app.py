import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Carrega os dados
df = pd.read_csv('diabetes.csv')

# Separa entrada e saída
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Treina o modelo
modelo = RandomForestClassifier()
modelo.fit(X, y)

# Título do app
st.title('🔬 Preditor de Diabetes')

st.markdown('Insira os dados abaixo para saber se a pessoa tem chance de ter diabetes:')

# Inputs do usuário
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Gravidez (Pregnancies)', 0, 20, 1)
    glucose = st.number_input('Glicose (Glucose)', 0, 200, 100)
    blood_pressure = st.number_input('Pressão arterial (BloodPressure)', 0, 140, 70)
    skin_thickness = st.number_input('Espessura da pele (SkinThickness)', 0, 100, 20)

with col2:
    insulin = st.number_input('Insulina (Insulin)', 0, 900, 80)
    bmi = st.number_input('IMC (BMI)', 0.0, 70.0, 25.0)
    dpf = st.number_input('DiabetesPedigreeFunction', 0.0, 3.0, 0.5)
    age = st.number_input('Idade', 0, 120, 33)

# Quando clicar em prever
if st.button('🔍 Prever'):
    entrada = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]],
                           columns=X.columns)

    pred = modelo.predict(entrada)

    if pred[0] == 1:
        st.error('⚠️ Possível caso de diabetes.')
    else:
        st.success('✅ Pouca chance de diabetes.')