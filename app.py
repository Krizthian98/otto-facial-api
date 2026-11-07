from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from deepface import DeepFace

app = Flask(__name__)

# Archivos de referencia de tu esposa
archivos_referencia = ["esposa_1.jpg", "esposa_2.jpg", "esposa_3.jpg"]

print("Verificando rostros de referencia...")
for archivo in archivos_referencia:
    if os.path.exists(archivo):
        print(f"-> Encontrado: {archivo}")
    else:
        print(f"-> Advertencia: No se encontró {archivo}")

@app.route('/reconocer', methods=['POST'])
def reconocer_rostro():
    if 'foto' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files['foto']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Guardar temporalmente la foto capturada por la ESP32-CAM
    temp_path = "temp_captura.jpg"
    cv2.imwrite(temp_path, img)

    reconocido = False
    mensaje = "Rostro no coincide"

    # Comparamos la captura contra cada una de las fotos de referencia de tu esposa
    try:
        for archivo in archivos_referencia:
            if os.path.exists(archivo):
                # DeepFace compara la foto nueva con la de referencia usando el modelo VGG-Face
                resultado = DeepFace.verify(
                    img1_path = temp_path, 
                    img2_path = archivo, 
                    model_name = "VGG-Face", 
                    enforce_detection = False
                )
                
                if resultado["verified"]:
                    reconocido = True
                    mensaje = "¡Esposa detectada!"
                    break
    except Exception as e:
        print(f"Error en análisis facial: {str(e)}")

    # Borrar la foto temporal para liberar espacio en el servidor
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify({"reconocido": reconocido, "mensaje": mensaje}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
