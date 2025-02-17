# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:07:28 2025

@author: riktome
"""

import streamlit as st

st.title("Evaluación de Apego y Creencias Negativas para EMDR")

# 📌 Formularios para ingresar datos
apego_ansioso = st.slider("Apego Ansioso", 0.0, 10.0, 5.0)
apego_evitativo = st.slider("Apego Evitativo", 0.0, 10.0, 5.0)
creencia_culpa = st.slider("Creencia de Culpa", 0.0, 10.0, 5.0)
creencia_indignidad = st.slider("Creencia de Indignidad", 0.0, 10.0, 5.0)
sistema_defensa = st.slider("Sistema de Defensa", 0.0, 10.0, 5.0)

# 📌 Crear DataFrame del paciente
df_patient = pd.DataFrame([{
    "Apego_Ansioso": apego_ansioso,
    "Apego_Evitativo": apego_evitativo,
    "Creencia_Culpa": creencia_culpa,
    "Creencia_Indignidad": creencia_indignidad,
    "Sistema_Defensa": sistema_defensa
}])

# 📌 Predicción del modelo
if st.button("Calcular Malestar Emocional"):
    y_pred = model.predict(df_patient)
    st.write(f"**Predicción de Malestar Emocional:** {y_pred[0]:.2f}")

    # 📊 Gráfico de barras
    st.bar_chart(df_patient.T)
