import PropsSpeckle as ps

import os

## Para verificar el directorio de trabajo, donde se guarda bitaora.txt
print("Directorio actual:", os.getcwd())

# Import the PropsSpeckle module
espacio = ps.PropsSpeckle()
# Parámetros del módulo
print('Versión:', espacio.version)
print('Nombre:', espacio.name)
print('Ruta',espacio.primera_linea)  # None

#Se llama una imagen, en este caso se da el nombre
espacio.imagen('4-3.tif',show=True)
# Información de estadística primer orden
espacio.statisticspro()
# Se normaliza la imagen
espacio.normalizar()
#Se vuelven a determinar los parámetros estadísticos de primer orden
espacio.statisticspro()
#Grafica el histograma
espacio.histograma()

#Se llama otra imagen, sin dar el nombre de la imagen,se grafica histograma
espacio.imagen(show=None)
espacio.histograma()