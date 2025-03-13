# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import logging
import os

import sys

# Configurar logging con salida a archivo y consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bitacora.txt",mode='w'),  # Guardar en archivo
        #logging.StreamHandler(sys.stdout)  # Imprimir en consola
    ]
)




class PropsSpeckle:

    # Class attribute
    version = '0.0.1.0'
    name = 'Caracterización de difusores'

    # Instance attributes
    def __init__(self, archivo="ruta.txt"):
        # ruta.txt es el archivo en la carpeta Speckle que contiene la ruta de la carpeta
        # con imágenes 
        try:
            ruta_archivo = os.path.join(os.path.dirname(__file__), "ruta.txt")
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                # Leer la primera línea y quitar espacios en blanco
                self.primera_linea = f.readline().strip()  
                self.primera_linea = os.path.join(os.path.dirname(__file__), self.primera_linea) #garantiza que la dirección esté completa
                logging.info(f"Se crea una instancia de PropSpeckle en la ruta {ruta_archivo}")
        except FileNotFoundError:
            print(f"Error: El archivo '{ruta_archivo}' no se encontró.")
            self.primera_linea = None
            sys.exit()
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")
            self.primera_linea = None
            sys.exit()


## Métodos de la clase PropsSpeckle ##


# Presentación en pantalla de la imagen
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
            logging.info(f"El nombre de la imagen de trabajo es: {imagen}")
        except FileNotFoundError:
            print(f"Error: La imagen '{ruta_imagen}' no se encontró.")
        except Exception as e:
            print(f"Ocurrió un error al abrir la imagen: {e}")


# Cálculo de parámetros estadísticos
    def statistics(self):
        # Calcula la media, varianza y la desviación estándar de la imagen
        self.media = np.mean(self.spec)
        self.desviacion = np.std(self.spec)
        self.varianza = np.var(self.spec)
        logging.info(f"Se ejecuta el método statistics")
        logging.info(f"Valor promedio de la imagen = {self.media}")
        logging.info(f"Valor desviación estándar de la imagen {self.desviacion}")
        logging.info(f"Valor varianza de la imagen {self.varianza}")


# Normalización de la imagen
    def normalizar(self):
        # Normaliza los  valores de intensidad de la imagen y actualiza
        # sus medeidas estadísticas.
        self.spec = self.spec / np.max(self.spec)
        self.media = np.mean(self.spec)
        self.desviacion = np.std(self.spec)
        self.varianza = np.var(self.spec)


# Histograma de la imagen
    def histograma(self):
        # Genera un histograma de la imagen
        plt.hist(self.spec.flatten(), bins=256, density=True)
        plt.title('Histograma de la imagen')
        plt.xlabel('Niveles de gris')
        plt.ylabel('Frecuencia')
        plt.show()
