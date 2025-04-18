# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import expon
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
        # Inicialización de variables
        self.spec = None
        self.media = None
        self.desviacion = None
        self.varianza = None
        self.constraste = None
        self.shape = None
        self.imagef = None
        self.mediaf = None
        self.desviacionf = None
        self.varianzaf = None
        self.constrastef = None




## Métodos de la clase PropsSpeckle ##

# Presentación en pantalla de la imagen
    def imagen(self, nombre:str=None, show=True):
        # Carga y muestra una imagen en escala de grises desde la ruta 
        # almacenada en el archivo de texto.
        while not nombre:
            nombre = input("Por favor, ingrese el nombre de la imagen con extensión: ").strip()
        
        try:
            ruta_imagen = self.primera_linea + '\\' + nombre 
            #Imagen en escala de grises
            imagen = Image.open(ruta_imagen).convert('L') 
            # Con self.spec se guarda la imagen en un arreglo de numpy para procesarla
            self.spec = np.array(imagen)
            #Muestra la imagen
            self.shape = self.spec.shape
            logging.info(f"Se ejecuta el método imagen, leyendo la imagen {nombre}")
            logging.info(f"El tamaño de la imagen es: {self.shape}")
            if show:
                imagen.show()  

            logging.info(f"El nombre de la imagen de trabajo es: {imagen}")
        except FileNotFoundError:
            print(f"Error: La imagen '{ruta_imagen}' no se encontró.")
        except Exception as e:
            print(f"Ocurrió un error al abrir la imagen: {e}")


# Cálculo de parámetros estadísticos
    def statisticspro(self):

        # Calcula la media, varianza y la desviación estándar de la imagen
        self.media = np.mean(self.spec)
        self.desviacion = np.std(self.spec)
        self.varianza = np.var(self.spec)
        self.constraste = self.desviacion / self.media

        logging.info(f"Se ejecuta el método statisticspro")
        logging.info(f"Valor promedio de la imagen = {self.media}")
        logging.info(f"Valor desviación estándar de la imagen {self.desviacion}")
        logging.info(f"Valor varianza de la imagen {self.varianza}")
        logging.info(f"Valor contraste de la imagen {self.constraste}")

        print(f"Valor de la intensidad media es = {self.media}")
        print(f"Valor de la desviación estándar es = {self.desviacion}")
        print(f"Valor de la varianza es = {self.varianza}")
        print(f"Valor del contraste es = {self.constraste}")


    def cal_media(self):
        # Calcula la media de la imagen
        self.media = np.mean(self.spec)
        logging.info(f"Se ejecuta el método cal_media")
        logging.info(f"Valor promedio de la imagen = {self.media}")
        return self.media
    
    def cal_desviacion(self):
        # Calcula la desviación estándar de la imagen
        self.desviacion = np.std(self.spec)
        logging.info(f"Se ejecuta el método cal_desviacion")
        logging.info(f"Valor desviación estándar de la imagen {self.desviacion}")
        return self.desviacion
    
    def cal_varianza(self):
        # Calcula la varianza de la imagen
        self.varianza = np.var(self.spec)
        logging.info(f"Se ejecuta el método cal_varianza")
        logging.info(f"Valor varianza de la imagen {self.varianza}")
        return self.varianza
    
    def cal_contraste(self):
        # Calcula el contraste de la imagen
        self.constraste = self.desviacion / self.media
        logging.info(f"Se ejecuta el método cal_contraste")
        logging.info(f"Valor contraste de la imagen {self.constraste}")
        return self.constraste

# Normalización de la imagen
    def normalizar(self):
        # Normaliza los  valores de intensidad de la imagen y actualiza
        # sus medeidas estadísticas.
        self.spec = self.spec / np.max(self.spec)
        logging.info(f"Se ejecuta el método normalizar")
        
# Histograma de la imagen
    def histograma(self):
        # Genera un histograma de la imagen
        plt.hist(self.spec.flatten(), bins=256, density=True)
        plt.title('Histograma de la imagen')
        plt.xlabel('Niveles de gris')
        plt.ylabel('Frecuencia')
        plt.show()
# Estadísticas de segundo orden
    def autocorrelacion(self, show=True, dim = 1, pixel_size=5.2):
        if self.spec is None:
            raise ValueError("La imagen no ha sido cargada. Por favor, carga una imagen primero.")
        # Calcula la autocorrelación de la imagen
        # Pixel_size es el tamaño del píxel en micrómetros.
        # Dim es la dimensión de la autocorrelación (1 o 2)
        # show indica si se debe mostrar la autocorrelación o no.
        f = np.fft.fft2(self.spec)  # Transformada de Fourier
        f_conj = np.conj(f)     # Conjugado complejo
        acorr = np.fft.ifft2(f * f_conj)  # Autocorrelación inversa
        acorr = np.fft.fftshift(acorr)  # Centrar la autocorrelación
        H, W = self.shape  # Dimensiones de la imagen
        x = np.linspace(-W//2, W//2, W) * pixel_size  # En micrómetros
        y = np.linspace(-H//2, H//2, H) * pixel_size  # En micrómetros

        if show:
            if dim == 1:
                # Mostrar la autocorrelación en una dimensión
                y = acorr[H//2,:]
                plt.plot(x, y, c='black')
                plt.scatter(x, y, s=8, c='red')
                plt.title('Autocorrelación')
                plt.xlabel('Desplazamiento en x (µm)')
                plt.ylabel('Autocorrelación')
                plt.show()
            elif dim == 2:
                # Mostrar la autocorrelación en dos dimensiones
                plt.imshow(np.abs(acorr), extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
                plt.title('Autocorrelación')
                plt.xlabel("Desplazamiento en x (µm)")
                plt.ylabel("Desplazamiento en y (µm)")
                plt.title("Autocorrelación en unidades de longitud")
                plt.colorbar()
                plt.show()
        logging.info(f"Se ejecuta el método autocorrelacion")
        return acorr
    def modificar(self):
        # Modifica la imagen de forma 
        # conveniente para analilzar la densidad de probabilidad de la muestra tomada.
        # Se ubica la moda como el nuevo valor de referencia y se resta a la imagen original.
        if self.spec is None:
            raise ValueError("La imagen no ha sido cargada. Por favor, carga una imagen primero.")
        # Obtener el valor más frecuente (moda)
        valores, cuentas = np.unique(self.spec, return_counts=True)
        moda = valores[np.argmax(cuentas)]
        # Crear una máscara booleana para los valores mayores o iguales a la moda
        mascara = self.spec >= moda
        
        # Crear imagen filtrada con NaNs por defecto (tipo float64)

        #Crea un array de NAN con la misma forma que la imagen original
        self.imagef = np.full(self.shape, np.nan, dtype=np.float64)
        # Asigna los valores de la imagen original restando la moda a la imagen filtrada 
        # donde la máscara es True, donde es False se queda como NaN
        self.imagef[mascara] = self.spec[mascara] - moda

        # Actualiza las estadísticas de primer orden de la imagen filtrada
        self.mediaf = np.nanmean(self.imagef)
        self.desviacionf = np.nanstd(self.spec)
        self.varianzaf = np.nanvar(self.spec)
        self.constrastef = self.desviacionf / self.mediaf


        #crea un array de valores distintos a NaN para graficar el histograma
        valores_validos = self.imagef[~np.isnan(self.imagef)].flatten()
        # Crear histograma normalizado (como una densidad de probabilidad)
        conteo, bins, _= plt.hist(valores_validos, bins=256, density=True, alpha=0.5, color='gray', label='Datos')

        # Eje x para la función teórica
        x = np.linspace(0, np.max(valores_validos), 1000)

        # Distribución exponencial teórica
        pdf_expon = expon(scale=self.mediaf).pdf(x)

        # Grafica de ambas
        plt.plot(x, pdf_expon, 'r-', label='Distribución exponencial teórica')
        plt.title('Comparación con distribución de speckle completamente polarizado')
        plt.xlabel('Intensidad (ajustada)')
        plt.ylabel('Densidad de probabilidad')
        plt.legend()
        plt.grid(True)
        plt.show()
    def autocorrelacionf(self, show=True, dim = 1, pixel_size=5.2):
        # Calcula la autocorrelación de la imagen filtrada
        # Pixel_size es el tamaño del píxel en micrómetros.
        # Dim es la dimensión de la autocorrelación (1 o 2)
        # show indica si se debe mostrar la autocorrelación o no.
        if self.spec is None or self.imagef is None:
            # Verifica si la imagen original o la imagen filtrada no han sido cargadas
            raise ValueError("La imagen no ha sido cargada o modificada. Por favor, carga una imagen primero.")
        # Los valores NaN se reemplazan por 0.0 para evitar problemas en la FFT
        # Pero se crea una máscara para no perder la información de la imagen original
        # La idea Luego dividir las autocorrelaciones para quitar el efecto de la máscara de 0s
        
        # Se cambian Nans por 0.0 para evitar problemas en la FFT
        imagen_temp = np.nan_to_num(self.imagef, nan=0.0)
        # Se crea una máscara para los valores originales
        # La máscara es 1 donde la imagen original no es NaN y 0 donde sí lo es
        mask = ~np.isnan(self.imagef)
        mask = mask.astype(float)

        # FFT de la imagen y la máscara
        fft_img = np.fft.fft2(imagen_temp)
        fft_mask = np.fft.fft2(mask)

        # Autocorrelación de la imagen y de la máscara
        autocorr_img = np.fft.fftshift(np.real(np.fft.ifft2(fft_img * np.conj(fft_img))))
        autocorr_mask = np.fft.fftshift(np.real(np.fft.ifft2(fft_mask * np.conj(fft_mask))))

        # Normalizar (evitando división por cero)
        autocorr_norm = np.divide(
            autocorr_img,
            autocorr_mask,
            out=np.zeros_like(autocorr_img),
            where=autocorr_mask > 0
        )

        # Mostrar
        
        if show:
            H, W = self.shape  # Dimensiones de la imagen
            x = np.linspace(-W//2, W//2, W) * pixel_size  # En micrómetros
            y = np.linspace(-H//2, H//2, H) * pixel_size  # En micrómetros
            if dim == 1:
                # Mostrar la autocorrelación en una dimensión
                y = autocorr_norm[H//2,:]
                plt.plot(x, y, c='black')
                plt.scatter(x, y, s=8, c='red')
                plt.title('Autocorrelación')
                plt.xlabel('Desplazamiento en x (µm)')
                plt.ylabel('Autocorrelación')
                plt.show()
            elif dim == 2:
                # Mostrar la autocorrelación en dos dimensiones
                plt.imshow(np.abs(autocorr_norm), extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
                plt.title('Autocorrelación')
                plt.xlabel("Desplazamiento en x (µm)")
                plt.ylabel("Desplazamiento en y (µm)")
                plt.title("Autocorrelación en unidades de longitud")
                plt.colorbar()
                plt.show()
        return autocorr_norm