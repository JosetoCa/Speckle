# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class PropsSpeckle:

    # Class attribute
    version = '0.0.1.0'
    name = 'Caracterización de difusores'

    # Instance attributes
    def __init__(self, archivo="ruta.txt"):
        # ruta.txt es el archivo en la carpeta Speckle que contiene la ruta de la carpeta
        # con imágenes 
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                # Leer la primera línea y quitar espacios en blanco
                self.primera_linea = f.readline().strip()  
        except FileNotFoundError:
            print(f"Error: El archivo '{archivo}' no se encontró.")
            self.primera_linea = None
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")
            self.primera_linea = None
    def imagen(self, show=True):
        # Carga y muestra una imagen en escala de grises desde la ruta 
        # almacenada en el archivo de texto.
        try:
            nombre = input("Ingrese el nombre de la imagen con su extensión: ")
            ruta_imagen = self.primera_linea + '\\' + nombre 
            #Imagen en escala de grises
            imagen = Image.open(ruta_imagen).convert('L') 
            # Con self.spec se guarda la imagen en un arreglo de numpy para procesarla
            self.spec = np.array(imagen)
            #Muestra la imagen
            if show:
                imagen.show()  
        except FileNotFoundError:
            print(f"Error: La imagen '{ruta_imagen}' no se encontró.")
        except Exception as e:
            print(f"Ocurrió un error al abrir la imagen: {e}")
    def statics(self):
        # Calcula la media, varianza y la desviación estándar de la imagen
        self.media = np.mean(self.spec)
        self.desviacion = np.std(self.spec)
        self.varianza = np.var(self.spec)
    def normalizar(self):
        # Normaliza los  valores de intensidad de la imagen y actualiza
        # sus medeidas estadísticas.
        self.spec = self.spec / np.max(self.spec)
        self.media = np.mean(self.spec)
        self.desviacion = np.std(self.spec)
        self.varianza = np.var(self.spec)
    def histograma(self):
        # Genera un histograma de la imagen
        plt.hist(self.spec.flatten(), bins=256, density=True)
        plt.title('Histograma de la imagen')
        plt.show()
