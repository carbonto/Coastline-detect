import cv2
import numpy as np


# Funcion para detectar la linea de costa utilizando Watershed
def detectar_linea_costa(frame):
    global sure_fg,sure_bg
    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral adaptativo
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Eliminar el ruido mediante operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Encontrar los marcadores para aplicar Watershed 
    # Se han variado los parametros de las iteraciones y la distancia multiplicada al maximo para obtener de manera correcta el background y el foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3* dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Aplicar Watershed para segmentar el área entre la arena y el mar
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [0, 255, 0]

    return frame

# Abrir el archivo de video
video_path = "video_linea.mp4"
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Cannot open video file")
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

    # Procesar el fotograma para detectar la línea entre la arena y el mar utilizando Watershed
    processed_frame = detectar_linea_costa(frame)

    # Escribir el fotograma procesado en el video de salida
    out.write(processed_frame)

    # Mostrar el video procesado en tiempo real
    cv2.imshow("Video Procesado", processed_frame)
    ####Debug
    # cv2.imshow("Thres", sure_fg)
    # cv2.imshow("Thr", sure_bg)
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los objetos utilizados
cap.release()
out.release()
cv2.destroyAllWindows()