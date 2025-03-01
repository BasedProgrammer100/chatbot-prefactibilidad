from fastapi import FastAPI
import pickle
import pdfkit
from pydantic import BaseModel

app = FastAPI()

# Cargar el modelo de ML
with open("modelo_viabilidad_mejorado.pkl", "rb") as file:
    modelo = pickle.load(file)

# Definir esquema de entrada
class Campamento(BaseModel):
    Capacidad: int
    Agua: int
    Energía: int
    Distancia_ciudad: int
    Accesibilidad: int
    Dias_operacion: int
    Costo_transporte: int
    Costo: int

# Endpoint para evaluar viabilidad
@app.post("/evaluar/")
def evaluar(campamento: Campamento):
    datos = [[
        campamento.Capacidad, campamento.Agua, campamento.Energía, 
        campamento.Distancia_ciudad, campamento.Accesibilidad, 
        campamento.Dias_operacion, campamento.Costo_transporte, campamento.Costo
    ]]
    
    resultado = modelo.predict(datos)[0]
    viabilidad = "Viable" if resultado == 1 else "No Viable"

    # Generar reporte en HTML
    html = f"""
    <h1>Reporte de Prefactibilidad</h1>
    <p><strong>Capacidad:</strong> {campamento.Capacidad} personas</p>
    <p><strong>Agua:</strong> {"Sí" if campamento.Agua else "No"}</p>
    <p><strong>Energía:</strong> {"Sí" if campamento.Energía else "No"}</p>
    <p><strong>Distancia a Ciudad:</strong> {campamento.Distancia_ciudad} km</p>
    <p><strong>Accesibilidad:</strong> {"Sí" if campamento.Accesibilidad else "No"}</p>
    <p><strong>Días de Operación:</strong> {campamento.Dias_operacion} días</p>
    <p><strong>Costo de Transporte:</strong> ${campamento.Costo_transporte}</p>
    <p><strong>Costo Estimado:</strong> ${campamento.Costo}</p>
    <p><strong>Viabilidad:</strong> {viabilidad}</p>
    """
    
    # Guardar reporte en PDF
    pdfkit.from_string(html, "reporte.pdf")

    return {"Viabilidad": viabilidad, "Reporte": "reporte.pdf"}
