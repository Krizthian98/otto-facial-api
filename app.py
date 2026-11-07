from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os

app = Flask(__name__)

# Cargar automáticamente todas las fotos de referencia disponibles
codificaciones_referencia = []
archivos_referencia = ["esposa_1.jpg", "esposa_2.jpg", "esposa_3.jpg"]

print("Cargando rostros de referencia...")
for archivo in archivos_referencia:
    if os.path.exists(archivo):
        img_ref = face_recognition.load_image_file(archivo)
        encs = face_recognition.face_encodings(img_ref)
        if len(encs) > 0:
            codificaciones_referencia.append(encs[0])
            print(f"-> Rostro cargado exitosamente desde: {archivo}")
        else:
            print(f"-> Advertencia: No se encontró un rostro válido en {archivo}")
    else:
        print(f"-> Archivo no encontrado: {archivo}")

if not codificaciones_referencia:
    print("¡ERROR CRÍTICO! No se cargó ninguna foto de referencia.")

@app.route('/reconocer', methods=['POST'])
def reconocer_rostro():
    if not codificaciones_referencia:
        return jsonify({"error": "No hay rostros de referencia cargados en el servidor"}), 500

    if 'foto' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files['foto']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Detectar rostros en la foto enviada por la ESP32-CAM
    codificaciones_recibidas = face_recognition.face_encodings(img)

    if len(codificaciones_recibidas) == 0:
        return jsonify({"reconocido": False, "mensaje": "No se detectó ningún rostro"}), 200

    # Compara el rostro recibido contra CUALQUIERA de las fotos de referencia
    rostro_actual = codificaciones_recibidas[0]
    coincidencias = face_recognition.compare_faces(codificaciones_referencia, rostro_actual, tolerance=0.5)

    if True in coincidencias:
        return jsonify({"reconocido": True, "mensaje": "¡Esposa detectada!"}), 200
    else:
        return jsonify({"reconocido": False, "mensaje": "Rostro no coincide"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)