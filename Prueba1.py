import PropsSpeckle as ps
import os

print("Directorio actual:", os.getcwd())

# Import the PropsSpeckle module
espacio = ps.PropsSpeckle()

print('Versión:', espacio.version)
print('Nombre:', espacio.name)
print('Ruta',espacio.primera_linea)  # None

espacio.imagen()
espacio.statistics()
print('Parámetros estadísticos de la imagen:')
print('Media:', espacio.media)
print('Desviación estándar:', espacio.desviacion)  
print('Varianza:', espacio.varianza) 
espacio.normalizar()
print('Parámetros estadísticos de la imagen normalizada:')
print('Media:', espacio.media)
print('Desviación estándar:', espacio.desviacion)  
print('Varianza:', espacio.varianza) 
espacio.histograma()