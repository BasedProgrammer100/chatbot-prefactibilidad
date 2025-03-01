import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generar dataset sintético
random.seed(42)
data = {
    "Capacidad": [random.randint(20, 300) for _ in range(100)],
    "Agua": [random.choice([0, 1]) for _ in range(100)],
    "Energía": [random.choice([0, 1]) for _ in range(100)],
    "Distancia_ciudad": [random.randint(5, 200) for _ in range(100)],  # km
    "Accesibilidad": [random.choice([0, 1]) for _ in range(100)],  # 1=Sí, 0=No
    "Dias_operacion": [random.randint(100, 365) for _ in range(100)],
    "Costo_transporte": [random.randint(5000, 50000) for _ in range(100)],
    "Costo": [random.randint(30000, 300000) for _ in range(100)],
}

df = pd.DataFrame(data)

# Definir si un campamento es viable
df["Viable"] = df.apply(lambda row: 1 if row["Agua"] and row["Energía"] and row["Accesibilidad"] else 0, axis=1)

# Separar variables predictoras y objetivo
X = df.drop(columns=["Viable"])
y = df["Viable"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

# Guardar el modelo
with open("modelo_viabilidad_mejorado.pkl", "wb") as file:
    pickle.dump(modelo, file)

# Evaluar modelo
accuracy = modelo.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Modelo guardado como 'modelo_viabilidad_mejorado.pkl'")
