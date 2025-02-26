import os
import numpy as np
from PIL import Image


imagenes = {}
carpeta_imagenes = os.path.dirname(__file__)

for i in range(0, 5):
    for j in range(1,4):
        
        #asignamos los nombres de las variables que contienen las imágenes
        var_name = 'speckle_' + str(i) + '_' + str(j)
        
        #nombres y ruta de las imágenes a leer
        nombre_archivo = f'{i}-{j}.tif'
        ruta_archivo = os.path.join(carpeta_imagenes, nombre_archivo)

        #leemos las imágenes y las guardamos en un diccionario
        imagen = Image.open(ruta_archivo).convert('L')
        imagenes[var_name] = np.array(imagen)

medias = []
varianzas = []
desviaciones = []
razones_md = []
razones_mv = []

for clave, valor in imagenes.items():
    media = np.mean(valor)
    varianza = np.var(valor)
    desviacion = np.std(valor)
    razon_md = media/desviacion
    razones_mv = varianza/media

    medias.append(f"la media del difusor {clave} es {media}")
    varianzas.append(f"la varianza del difusor {clave} es {varianza}")
    desviaciones.append(f"la desviación estándar del difusor {clave} es {desviacion}")
    razones_md.append(razon_md)
    razones_mv.append(razones_mv)

print(razones_md)
print(razones_mv)
# print(medias)