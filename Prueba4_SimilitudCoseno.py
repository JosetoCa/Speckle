import PropsSpeckle as ps


# Se crea el objeto espacio2 de la clase PropsSpeckle
espacio2 = ps.PropsSpeckle()
# Carga la imagen y calcula estadísticas y autocorrelación
espacio2.imagen('4-150-1.tif',show=False)
espacio2.statisticspro()
# Calcula la similitud coseno entre las imágenes
espacio2.histograma_similaridad()