import cv2
import numpy as np

# Función para detectar la línea entre la arena y el mar en un fotograma del video
def detectar_linea_costa(frame):
    # Convertir el fotograma a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir rangos de colores para la arena y el mar
    lower_sand = np.array([50, 50, 50])
    upper_sand = np.array([100, 255, 255])
    lower_sea = np.array([40, 50, 50])
    upper_sea = np.array([120, 200, 200])

    # Segmentar el fotograma en función de los rangos de colores
    mask_sand = cv2.inRange(hsv, lower_sand, upper_sand)
    mask_sea = cv2.inRange(hsv, lower_sea, upper_sea)

    # Unir las máscaras resultantes
    mask = cv2.bitwise_or(mask_sand, mask_sea)

    # Aplicar operaciones morfológicas para eliminar ruido y mejorar la detección
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar los contornos de la línea entre la arena y el mar
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en el fotograma original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    return frame

# Abrir el archivo de video
video_path = "video_linea.mp4"
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir el video")
    exit()

# Obtener el tamaño del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Crear el objeto para guardar el video procesado
out = cv2.VideoWriter("video_procesado.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Leer los fotogramas del video
while True:
    ret, frame = cap.read()

    # Verificar si se alcanzó el final del video
    if not ret:
        break

    # Procesar el fotograma para detectar la línea entre la arena y el mar
    processed_frame = detectar_linea_costa(frame)

    # Escribir el fotograma procesado en el video de salida
    out.write(processed_frame)

    # Mostrar el video procesado en tiempo real
    cv2.imshow("Video Procesado", processed_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
