import os
import mimetypes
import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher

# Configuración de Tesseract PSM (Page Segmentation Mode).
# Para placas, a veces PSM 6, 7 u 8 funcionan bien. Puedes experimentar.
PSM_VALUE = 8

# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------
def recortar_franja_central(imagen):
    alto, ancho = imagen.shape[:2]
    # Calcular el 20% de la altura de la imagen
    recorte_superior = int(alto * 0.18)
    recorte_inferior = int(alto * 0.18)

    recorte_izquierda = int(ancho * 0.10)
    recorte_derecha = int(ancho * 0.10)

    # Recortar la imagen: eliminar el 20% superior e inferior
    roi = imagen[recorte_superior:alto - recorte_inferior, 0+recorte_izquierda:ancho-recorte_derecha]
    return roi

def text_similarity(text1, text2):
    """
    Calcula la similitud (entre 0.0 y 1.0) entre dos textos
    usando SequenceMatcher de difflib.
    """
    return SequenceMatcher(None, text1, text2).ratio()

def limpiar_texto(texto):
    """
    Convierte a mayúsculas, elimina espacios y filtra solo caracteres
    válidos para placas de Costa Rica (A-Z, 0-9 y guión).
    """
    texto = texto.upper().strip()
    permitido = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    texto_filtrado = "".join([c for c in texto if c in permitido])
    return texto_filtrado

def reconocer_texto(imagen):
    """
    Aplica Tesseract OCR a una imagen, usando la configuración:
      --oem 3, --psm PSM_VALUE, y whitelist de caracteres.
    """
    config_tesseract = (
        f"--oem 3 --psm {PSM_VALUE} "
        + "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    )
    texto = pytesseract.image_to_string(imagen, config=config_tesseract)
    return limpiar_texto(texto)

# -----------------------------------------------------------------------------
# PREPROCESADORES PARA IMAGEN INICIAL
# -----------------------------------------------------------------------------
def preprocesado_basico(imagen):
    """
    1. Escala de grises
    2. Filtro bilateral
    3. Umbral adaptativo gaussiano
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    filtrado = cv2.bilateralFilter(gris, 9, 75, 75)
    umbral = cv2.adaptiveThreshold(
        filtrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return umbral

def preprocesado_otsu(imagen):
    """
    1. Escala de grises
    2. Filtro bilateral
    3. Umbralización con Otsu
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    filtrado = cv2.bilateralFilter(gris, 9, 75, 75)
    _, umbral = cv2.threshold(filtrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return umbral

def preprocesado_morfologico(imagen):
    """
    1. Escala de grises
    2. Umbral con Otsu
    3. Operaciones morfológicas (opening y closing) para limpiar ruido
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binario = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing

def preprocesado_resize(imagen):
    """
    1. Redimensionar la imagen (2x)
    2. Filtro bilateral
    3. Umbral Otsu
    """
    h, w = imagen.shape[:2]
    imagen_resize = cv2.resize(imagen, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

    gris = cv2.cvtColor(imagen_resize, cv2.COLOR_BGR2GRAY)
    filtrado = cv2.bilateralFilter(gris, 9, 75, 75)
    _, umbral = cv2.threshold(filtrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return umbral

def preprocesado_gaussiano(imagen):
    """
    1. Escala de grises
    2. Desenfoque Gaussiano
    3. Umbral Otsu
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, umbral = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return umbral

# -----------------------------------------------------------------------------
# PREPROCESADORES PARA IMAGEN FINAL (recortada/enderezada)
#    - Se pueden repetir, pero con ligeras variaciones en parámetros
# -----------------------------------------------------------------------------
def preprocesado_basico_final(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    filtrado = cv2.bilateralFilter(gris, 11, 17, 17)
    umbral = cv2.adaptiveThreshold(
        filtrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 13, 5
    )
    return umbral

def preprocesado_otsu_final(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    filtrado = cv2.bilateralFilter(gris, 11, 17, 17)
    _, umbral = cv2.threshold(filtrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return umbral

def preprocesado_morfologico_final(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binario = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing

def preprocesado_resize_final(imagen):
    h, w = imagen.shape[:2]
    imagen_resize = cv2.resize(imagen, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

    gris = cv2.cvtColor(imagen_resize, cv2.COLOR_BGR2GRAY)
    filtrado = cv2.bilateralFilter(gris, 11, 17, 17)
    _, umbral = cv2.threshold(filtrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return umbral

def preprocesado_gaussiano_final(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, umbral = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return umbral

# -----------------------------------------------------------------------------
# FUNCIONES DE DETECCIÓN / RECORTE DE PLACA (OPCIONAL)
# -----------------------------------------------------------------------------
def encontrar_contorno_placa(imagen_binaria):
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    placa_contorno = None
    max_area = 0
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        area = cv2.contourArea(approx)
        if area > max_area:
            max_area = area
            placa_contorno = approx
    return placa_contorno

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformar_perspectiva(imagen, contorno):

    try:
        pts = contorno.reshape(4, 2)
    except Exception as err:
        print(f"Error en transformar_perspectiva: {err}")
        return imagen

    rect = ordenar_puntos(pts)

    anchoA = np.linalg.norm(rect[2] - rect[3])
    anchoB = np.linalg.norm(rect[1] - rect[0])
    max_ancho = max(int(anchoA), int(anchoB))

    altoA = np.linalg.norm(rect[1] - rect[2])
    altoB = np.linalg.norm(rect[0] - rect[3])
    max_alto = max(int(altoA), int(altoB))

    dst = np.array([
        [0, 0],
        [max_ancho - 1, 0],
        [max_ancho - 1, max_alto - 1],
        [0, max_alto - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imagen, M, (max_ancho, max_alto))
    return warped

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    carpeta_imagenes = "cuban-plates-dataset"

    # Definimos 5 preprocesadores para la imagen inicial
    preprocessors_initial = {
        "BASICO": preprocesado_basico,
        "OTSU": preprocesado_otsu,
        "MORFOLOGICO": preprocesado_morfologico,
        "RESIZE": preprocesado_resize,
        "GAUSSIANO": preprocesado_gaussiano
    }

    # Definimos 5 preprocesadores para la imagen final (recortada/enderezada)
    preprocessors_final = {
        "BASICO_FINAL": preprocesado_basico_final,
        "OTSU_FINAL": preprocesado_otsu_final,
        "MORFO_FINAL": preprocesado_morfologico_final,
        "RESIZE_FINAL": preprocesado_resize_final,
        "GAUSS_FINAL": preprocesado_gaussiano_final
    }

    # Recorremos todos los archivos de la carpeta
    for nombre_archivo in os.listdir(carpeta_imagenes):
        ruta_completa = os.path.join(carpeta_imagenes, nombre_archivo)

        # Verificamos si es un archivo de imagen
        mime_type, _ = mimetypes.guess_type(ruta_completa)
        if not (mime_type and mime_type.startswith("image")):
            continue  # No es imagen, se omite

        # Texto esperado: nombre del archivo sin extensión (en mayúsculas)
        texto_esperado = os.path.splitext(nombre_archivo)[0].upper()

        # Cargamos la imagen original
        imagen_original = cv2.imread(ruta_completa)
        if imagen_original is None:
            print(f"[ERROR] No se pudo cargar la imagen: {nombre_archivo}")
            continue

        # ---------------------------------------------------------------------
        # PRIMERO, PROBAMOS PREPROCESADORES EN LA IMAGEN INICIAL (SIN RECORTAR)
        # ---------------------------------------------------------------------
        mejor_texto_inicial = None
        mejor_sim_inicial = 0.0
        mejor_metodo_inicial = None

        for nombre_metodo, funcion in preprocessors_initial.items():

            # aqui la recortamos desde el inicio
            imagen_recortada = recortar_franja_central(imagen_original)

            # Preprocesar
            img_preproc = funcion(imagen_recortada)
            # OCR
            texto_inferido = reconocer_texto(img_preproc)
            print(funcion , "->" , texto_inferido)
            # Similaridad
            sim = text_similarity(texto_inferido, texto_esperado)

            if sim > mejor_sim_inicial:
                mejor_sim_inicial = sim
                mejor_texto_inicial = texto_inferido
                mejor_metodo_inicial = nombre_metodo

        # ---------------------------------------------------------------------
        # DETECTAMOS Y RECORTAMOS LA PLACA (O HACEMOS PERSPECTIVA)
        # ---------------------------------------------------------------------
        # Podemos usar algún preprocesado simple para contornos
        img_bin = preprocesado_otsu(imagen_original)  # O el que prefieras
        contorno_placa = encontrar_contorno_placa(img_bin)

        if contorno_placa is not None:
            imagen_placa = transformar_perspectiva(imagen_original, contorno_placa)
        else:
            # Si no se encontró contorno, usar la original como "final"
            imagen_placa = imagen_original

        # ---------------------------------------------------------------------
        # LUEGO, PROBAMOS PREPROCESADORES EN LA IMAGEN FINAL (RECORTADA)
        # ---------------------------------------------------------------------
        mejor_texto_final = None
        mejor_sim_final = 0.0
        mejor_metodo_final = None

        for nombre_metodo, funcion in preprocessors_final.items():
            img_preproc_final = funcion(imagen_placa)
            texto_inferido_final = reconocer_texto(img_preproc_final)
            sim_final = text_similarity(texto_inferido_final, texto_esperado)

            print(funcion, "->", texto_inferido_final)

            if sim_final > mejor_sim_final:
                mejor_sim_final = sim_final
                mejor_texto_final = texto_inferido_final
                mejor_metodo_final = nombre_metodo

        # ---------------------------------------------------------------------
        # COMPARAR RESULTADOS: INICIAL vs FINAL
        # ---------------------------------------------------------------------
        # Elegimos el que tenga mayor similitud global
        if mejor_sim_inicial >= mejor_sim_final:
            print(f"\n[IMAGEN: {nombre_archivo}]")
            print(f"  > Mejor método (INICIAL): {mejor_metodo_inicial}")
            print(f"  > Texto inferido: {mejor_texto_inicial}")
            print(f"  > Texto esperado: {texto_esperado}")
            print(f"  > Similitud: {mejor_sim_inicial:.3f}")
        else:
            print(f"\n[IMAGEN: {nombre_archivo}]")
            print(f"  > Mejor método (FINAL): {mejor_metodo_final}")
            print(f"  > Texto inferido: {mejor_texto_final}")
            print(f"  > Texto esperado: {texto_esperado}")
            print(f"  > Similitud: {mejor_sim_final:.3f}")

if __name__ == "__main__":
    main()
