## Prueba para verificar el uso de la Prueba de Bondad ##

import PropsSpeckle as ps
import numpy as np
import matplotlib.pyplot as plt
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
espacio.imagen('0-3.tif',show=True)
espacio.pruebaBondad()


