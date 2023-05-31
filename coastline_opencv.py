import cv2
import numpy as np

# Función para detectar la línea entre la arena y el mar en una imagen
def detectar_linea_costa(image):
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos de colores para la arena y el mar
    lower_sand = np.array([50, 50, 50])
    upper_sand = np.array([100, 255, 255])
    lower_sea = np.array([40, 50, 50])
    upper_sea = np.array([120, 200, 200])

    # Segmentar la imagen en función de los rangos de colores
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

    # Dibujar los contornos en la imagen original
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    return image

# Cargar la imagen
image_path = "awawa.jpg"
image = cv2.imread(image_path)

# Redimensionar la imagen si es necesario para una mejor visualización
# image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Detectar la línea entre la arena y el mar en la imagen
processed_image = detectar_linea_costa(image)

# Mostrar la imagen procesada
cv2.imshow("Imagen", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
