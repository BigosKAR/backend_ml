import os
import joblib
import json
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Load the model once at module level (not every time function is called)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'best_xgb_model.pkl')
preprocess_path = os.path.join(base_dir, 'preprocessing_config.pkl')
data_path = os.path.join(base_dir, 'db_computers_2025_clean.csv')

model = joblib.load(model_path)
preprocessing = joblib.load(preprocess_path)
expected_columns = preprocessing['columns']
imputer = preprocessing['imputer']

# Load full dataset for recommendations
df = pd.read_csv(data_path)

# Columns to use for nearest neighbors — tune this list as you want
selected_columns = ['Pantalla_Tamaño de la pantalla', 'Pantalla_Tecnología de la pantalla', 
                    'Pantalla_Luminosidad', 'Procesador_Procesador', 'Disco duro_Tipo de disco duro', 'Gráfica_Salida de vídeo', 
                    'Conectividad_Lector de tarjetas', 'Comunicaciones_Versión Bluetooth', 'Cámara_Función de la cámara', 'Sonido_Número de altavoces', 
                    'Alimentación_Batería', 'Alimentación_Autonomía de la batería', 'Alimentación_Vatios-hora', 'Alimentación_Alimentación', 
                    'Medidas y peso_Material de la carcasa', 'Medidas y peso_Diseño', 'Medidas y peso_Profundidad', 'Medidas y peso_Ancho', 'Medidas y peso_Peso', 
                    'Comunicaciones_Estándar wifi', 'Disco duro_Número de discos duros (instalados)', 'Gráfica_Memoria gráfica', 'Sistema operativo_Sistema operativo', 
                    'Sonido_Sistema de altavoces', 'Pantalla_Formato de imagen', 'Procesador_Caché', 'RAM_Memoria RAM', 'RAM_Tipo de RAM', 'Disco duro_Capacidad de memoria SSD', 
                    'Alimentación_Número de celdas', 'Procesador_Número de núcleos del procesador', 'Procesador_Frecuencia de reloj', 'Procesador_Frecuencia turbo máx.', 'Gráfica_Tipo de memoria gráfica', 'Gráfica_Tarjeta gráfica', 'Almacenamiento_Lector óptico', 'Pantalla_Pantalla', 
                    'Comunicaciones_Estándar LAN', 'Tipo', 'Propiedades de la carcasa_Tipo de caja', 'Medidas y peso_Alto', 'Propiedades de la carcasa_Alimentación', 'Procesador_Zócalo de CPU', 'Procesador_Número de hilos de ejecución', 
                    'Procesador_TDP', 'product_type_group', 'is_laptop', 'marca', 'screen_size_class', 'screen_width_px', 'screen_height_px', 'screen_total_px', 'screen_resolution_class', 'SSD_missing', 'Disco duro_Numero de discos', 
                    'Comunicaciones_bluetooth', 'Comunicaciones_lan', 'Comunicaciones_missing', 'Comunicaciones_nfc', 'Comunicaciones_infrarrojos', 'Comunicaciones_wifidirect', 'Comunicaciones_ethernet', 'has_webcam', 'GPU_missing', 
                    'has_webcam_from_equipamiento', 'años_desde_lanzamiento', 'release_year_missing']

# Precompute subset + encodings + imputing + scaling for recommendation
subset_df = df[selected_columns].copy()
subset_encoded = pd.get_dummies(subset_df).fillna(0)

# Fit imputer and scaler once
rec_imputer = SimpleImputer(strategy='mean')
subset_imputed = pd.DataFrame(rec_imputer.fit_transform(subset_encoded), columns=subset_encoded.columns)

rec_scaler = StandardScaler()
subset_scaled = rec_scaler.fit_transform(subset_imputed)

# Initialize SHAP explainer once
explainer = shap.TreeExplainer(model)

with open(os.path.join(base_dir, 'feature_defaults_by_tipo.json'), 'r', encoding='utf-8') as f:
    default_values_by_tipo = json.load(f)

def merge_with_defaults(user_input: dict) -> dict:
    """
    Fill missing fields in user input using defaults specific to 'Tipo' (Laptop/Desktop).
    Falls back to the first available group if 'Tipo' not provided or unrecognized.
    """
    tipo = user_input.get("Tipo", "Laptop")  # Default to "Laptop" if not provided

    tipo_defaults = {}
    for col, tipo_dict in default_values_by_tipo.items():
        # Try to fetch value for given tipo; fallback to first one if not found
        if tipo in tipo_dict:
            tipo_defaults[col] = tipo_dict[tipo].get("default")
        else:
            fallback_tipo = next(iter(tipo_dict))
            tipo_defaults[col] = tipo_dict[fallback_tipo].get("default")

    # Merge defaults with user input (user input takes precedence)
    merged = {**tipo_defaults, **user_input}
    return merged


def preprocess_input(data: dict) -> pd.DataFrame:
    df_input = pd.DataFrame([data])

    # One-hot encode
    df_encoded = pd.get_dummies(df_input)

    # Align columns with training data
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)

    # Impute any missing values (as final safety net)
    df_imputed = pd.DataFrame(imputer.transform(df_encoded), columns=expected_columns)
    return df_imputed

def explain_prediction(preprocessed_df: pd.DataFrame, top_n=5):
    """Returns top N features contributing to the prediction with their SHAP values."""
    shap_values = explainer.shap_values(preprocessed_df)
    shap_dict = dict(zip(preprocessed_df.columns, shap_values[0]))
    top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    explanation = [{"feature": f, "contribution": float(val)} for f, val in top_features]
    return explanation

def recommend_nearest_partial_input(input_dict, top_n=5):
    # Prepare input dataframe with selected columns
    input_df = pd.DataFrame([input_dict])[selected_columns]

    # One-hot encode input and align columns with subset_encoded columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=subset_encoded.columns, fill_value=0)

    # Impute and scale with pre-fitted imputers and scalers
    input_imputed = pd.DataFrame(rec_imputer.transform(input_encoded), columns=subset_encoded.columns)
    input_scaled = rec_scaler.transform(input_imputed)

    # Compute distances and get nearest indices
    distances = euclidean_distances(input_scaled, subset_scaled)[0]
    nearest_indices = distances.argsort()[:top_n]

    # Return only Título and Precio_Rango columns from recommended rows
    return df.loc[nearest_indices, ['Título', 'Precio_Rango']].to_dict(orient='records')

def predict_price_with_explanation(user_input: dict) -> dict:
    """Predict price, provide feature importance, and recommend nearest neighbors."""
    full_input = merge_with_defaults(user_input)
    preprocessed = preprocess_input(full_input)
    log_price = model.predict(preprocessed)
    price = np.expm1(log_price)[0]

    explanation = explain_prediction(preprocessed)
    recommendations = recommend_nearest_partial_input(full_input)

    return {
        "predicted_price": round(price, 2),
        "feature_importance": explanation,
        "nearest_recommendations": recommendations
    }
