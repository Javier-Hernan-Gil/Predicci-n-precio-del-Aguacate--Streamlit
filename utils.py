# Paquetes
import streamlit as st
import sklearn,numpy
import pickle,joblib
from joblib import load
from sklearn.neural_network import MLPRegressor
from numpy import loadtxt


# Path del modelo preentrenado
MODEL_PATH = 'modelo_full.pkl'

def model_prediction(uploaded_file, model):

    modelo_cargado = joblib.load(MODEL_PATH) # Carga del modelo.
    archivo_cargado = loadtxt(uploaded_file, delimiter=',') # Carga el archivo
    archivo_ajustado = archivo_cargado.reshape(1,-1)
    preds = modelo_cargado.predict(archivo_ajustado)
    return preds

