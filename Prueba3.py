import numpy as np
import matplotlib.pyplot as plt
import PropsSpeckle as ps

seed = 42  # Semilla para reproducibilidad

np.random.seed(seed)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Parámetros
mu = 2.0
size = (256, 256)

# Generar imagen con distribución exponencial negativa escalada 1/mu
image = np.random.exponential(scale=1/mu, size=size)

# Normalizar al rango [0, 255]
image_normalized = image - image.min()
image_normalized /= image_normalized.max()
image_scaled = np.round(image_normalized * 255).astype(np.uint8) 

# Guardar imagen como PNG
img = Image.fromarray(image_scaled)
img.save('C:\\Proyectos\\Speckle\\images\\imagen_exponencial.png')

espacio1 =  ps.PropsSpeckle()
espacio1.imagen('imagen_exponencial.png',show=True)

espacio1.prueba(imagen = 'o')


espacio1.pruebaBondad(imagen = 'o')



