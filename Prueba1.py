import PropsSpeckle as ps

# Import the PropsSpeckle module
espacio = ps.PropsSpeckle()

print(espacio.primera_linea)  # None

espacio.imagen()
espacio.statics()
print(espacio.media, espacio.desviacion, espacio.varianza)
espacio.normalizar()
print(espacio.media, espacio.desviacion, espacio.varianza)
espacio.histograma()