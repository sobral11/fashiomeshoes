import os
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import argparse


def load_model_and_mapping(model_dir):
    model_path = os.path.join(model_dir, 'shoe_model_best.h5')
    model = tf.keras.models.load_model(model_path)

    mapping_file = os.path.join(model_dir, 'shoe_categories.json')

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    return model, mapping_data


def translate_category(category_name):
    translation_map = {
        'sneaker': 'casual sneaker',
        'casual': 'social shoe',
        'heel': 'heel',
        'boot': 'boot',
        'slide': 'slide'
    }
    return translation_map.get(category_name.lower(), category_name)


def preprocess_image(image_path, target_size=(224, 224)):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array * 255.0)
            return img_array
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None


def get_confidence_level(confidence):
    if confidence >= 0.65:
        return "ALTA CONFIANCA"
    elif confidence >= 0.50:
        return "CONFIANCA MODERADA"
    elif confidence >= 0.30:
        return "CONFIANCA BAIXA"
    else:
        return "CONFIANCA MUITO BAIXA - VERIFICAR"


def predict_single_image(model, mapping_data, image_path):
    processed_image = preprocess_image(image_path)

    if processed_image is None:
        return None

    input_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(input_image, verbose=0)

    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    index_to_category = mapping_data['index_to_category']
    predicted_class = index_to_category[str(predicted_class_idx)]

    translated_class = translate_category(predicted_class)
    confidence_level = get_confidence_level(confidence)

    return translated_class, confidence, confidence_level


def find_image_file(base_path):
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    if os.path.exists(base_path):
        return base_path

    for ext in extensions:
        test_path = base_path + ext
        if os.path.exists(test_path):
            return test_path

    if '.' not in os.path.basename(base_path):
        for ext in extensions:
            test_path = base_path + ext
            if os.path.exists(test_path):
                return test_path

    return None


def main():
    parser = argparse.ArgumentParser(description='Predição de uma única imagem')
    parser.add_argument('--image', required=True, help='Caminho da imagem para predição')
    parser.add_argument('--model_dir', default='grouped_shoe_models',
                        help='Diretório do modelo (padrão: grouped_shoe_models)')

    args = parser.parse_args()

    print("INICIANDO PREDIÇÃO...")
    print(f"Modelo: {args.model_dir}")
    print(f"Imagem: {args.image}")

    image_path = find_image_file(args.image)
    if not image_path:
        print(f"Imagem não encontrada: {args.image}")
        print("Tente especificar a extensão completa (ex: --image 'caminho/imgteste1.jpg')")
        return

    print(f"Imagem encontrada: {image_path}")

    try:
        model, mapping_data = load_model_and_mapping(args.model_dir)
        print("Modelo carregado com sucesso!")

        predicted_class, confidence, confidence_level = predict_single_image(model, mapping_data, image_path)

        if predicted_class:
            print("\n" + "=" * 40)
            print("RESULTADO DA PREDIÇÃO")
            print("=" * 40)
            print(f"Classe predita: {predicted_class}")
            print(f"Confiança: {confidence:.4f} ({confidence * 100:.2f}%)")
            print(f"Nivel de confiança: {confidence_level}")

        else:
            print("Erro ao processar a imagem")

    except Exception as e:
        print(f"Erro durante a predição: {e}")


if __name__ == "__main__":
    main()