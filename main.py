from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import base64
import os
import uvicorn

app = FastAPI(title="API de Classificação de Calçados")

model = None
mapping_data = None


class PredictionRequest(BaseModel):
    image: str


class PredictionResponse(BaseModel):
    predicted_class: str
    original_class: str
    confidence_level: str
    success: bool


def load_model():
    global model, mapping_data

    try:
        model_path = 'shoe_model_best.h5'
        mapping_path = 'shoe_categories.json'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        model = tf.keras.models.load_model(model_path)

        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        print("Modelo e mapeamento carregados com sucesso!")

    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        raise


def translate_category(category_name):
    translation_map = {
        'sneaker': 'Tênis Casual',
        'casual': 'Sapato Social',
        'heel': 'Salto',
        'boot': 'Bota',
        'slide': 'Slide/Chinelo'
    }
    return translation_map.get(category_name.lower(), category_name)


def get_confidence_level(confidence):
    if confidence >= 0.65:
        return "ALTA CONFIANÇA"
    elif confidence >= 0.40:
        return "CONFIANÇA MODERADA"
    elif confidence >= 0.20:
        return "CONFIANÇA BAIXA"
    else:
        return "CONFIANÇA MUITO BAIXA - VERIFICAR"


def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array * 255.0)
        return img_array
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "model_loaded": model is not None})


@app.post("/predict", response_model=PredictionResponse)
async def predict(request_data: PredictionRequest):
    global model, mapping_data

    if model is None or mapping_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        image_bytes = base64.b64decode(request_data.image)

        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            raise HTTPException(status_code=400, detail="Erro ao processar imagem")

        input_image = np.expand_dims(processed_image, axis=0)

        predictions = model.predict(input_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        predicted_class = mapping_data['index_to_category'][str(predicted_class_idx)]
        translated_class = translate_category(predicted_class)
        confidence_level = get_confidence_level(confidence)

        return PredictionResponse(
            predicted_class=translated_class,
            original_class=predicted_class,
            confidence_level=confidence_level,
            success=True
        )

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")


@app.get("/")
async def home():
    return JSONResponse(content={
        "message": "API de Classificação de Calçados",
        "endpoints": {
            "GET /health": "Health check da aplicação",
            "POST /predict": "Fazer predição de imagem (enviar JSON com 'image' em base64)"
        },
        "usage": {
            "example_request": {
                "image": "base64_encoded_image_string"
            }
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)