import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 📌 Función para ingresar datos manualmente
def input_patient_data():
    print("\nIngrese las respuestas del paciente en una escala de 0 a 10.")
    apego_ansioso = float(input("Apego Ansioso: "))
    apego_evitativo = float(input("Apego Evitativo: "))
    creencia_culpa = float(input("Creencia de Culpa: "))
    creencia_indignidad = float(input("Creencia de Indignidad: "))
    sistema_defensa = float(input("Sistema de Defensa: "))

    return pd.DataFrame([{
        "Apego_Ansioso": apego_ansioso,
        "Apego_Evitativo": apego_evitativo,
        "Creencia_Culpa": creencia_culpa,
        "Creencia_Indignidad": creencia_indignidad,
        "Sistema_Defensa": sistema_defensa
    }])

# 📌 Solicitar datos del paciente
df_patient = input_patient_data()

# 📌 Modelo de regresión previamente entrenado
np.random.seed(42)
X_train = np.random.uniform(0, 10, (100, 5))
y_train = 2.5 * X_train[:, 0] - 1.2 * X_train[:, 1] + 3.1 * X_train[:, 2] + 2.8 * X_train[:, 3] + 1.5 * X_train[:, 4] + np.random.normal(0, 2, 100)

model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Hacer predicción
y_pred = model.predict(df_patient)

# 📊 Mostrar resultado
print(f"\nPredicción de Malestar Emocional: {y_pred[0]:.2f}")

# 📌 Gráfica de dispersión con la predicción del paciente
plt.figure(figsize=(8, 6))
sns.regplot(x=X_train[:, 0], y=y_train, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
plt.scatter(df_patient["Apego_Ansioso"], y_pred, color='green', s=100, label="Paciente")
plt.title("Relación entre Apego Ansioso y Malestar Emocional")
plt.xlabel("Apego Ansioso")
plt.ylabel("Malestar Emocional")
plt.legend()
plt.show()
