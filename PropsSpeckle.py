# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import expon, linregress, chisquare, t as t_dist, chi2
import sys
import logging
import os
import pandas as pd
from scipy.ndimage import uniform_filter

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
        self.filtrada = None
        self.pixel_medio = None




## Métodos de la clase PropsSpeckle ##

### Estadísticas de segundo orden ###
    def autocorrelacion(self, imagen = 'o', show=True, dim = 1, pixel_size=5.2):
        if self.spec is None:
            raise ValueError("La imagen no ha sido cargada. Por favor, carga una imagen primero.")
        if imagen == 'o':
            print("Calculando la autocorrelación de la imagen original...")
            # Calcula la autocorrelación de la imagen
            # Pixel_size es el tamaño del píxel en micrómetros.
            # Dim es la dimensión de la autocorrelación (1 o 2)
            # show indica si se debe mostrar la autocorrelación o no.
            f = np.fft.fft2(self.spec)  # Transformada de Fourier
            f_conj = np.conj(f)     # Conjugado complejo
            acorr = np.fft.ifft2(f * f_conj)  # Autocorrelación inversa
            acorr = np.fft.fftshift(acorr)  # Centrar la autocorrelación

            # Encontrar el píxel máximo (pico central)
            max_pos = np.unravel_index(np.argmax(acorr), acorr.shape)
            y0, x0 = max_pos
            max_val = acorr[y0, x0]

            # Mínimo de la imagen
            min_val = acorr.min()

            # Valor medio entre máximo y mínimo
            mid_val = (max_val + min_val) / 2

            # Crear malla de distancias desde el centro del pico
            Y, X = np.indices(acorr.shape)
            R = np.sqrt((X - x0)**2 + (Y - y0)**2)
            R = R.astype(int)

            # Perfil radial promedio
            r_max = R.max()
            radial_profile = np.array([acorr[R == r].mean() for r in range(r_max)])

            # Buscar el radio donde cae por debajo del valor medio (puedes ajustar criterio)
            r_mid = np.argmax(radial_profile < mid_val)
            self.pixel_medio = r_mid  

            print(f"La intensidad cae al valor medio entre máximo y mínimo en un radio de {r_mid} píxeles.")

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
        if imagen == 'm':
            print("Calculando la autocorrelación de la imagen modificada...")
            # Calcula la autocorrelación de la imagen modificada
            # Calcula la autocorrelación de la imagen filtrada
            # Pixel_size es el tamaño del píxel en micrómetros.
            # Dim es la dimensión de la autocorrelación (1 o 2)
            # show indica si se debe mostrar la autocorrelación o no.
            if self.spec is None and self.imagef is None:
            # Verifica si la imagen original o la imagen filtrada no han sido cargadas
                raise ValueError("La imagen no ha sido cargada y modificada. Por favor, carga una imagen primero.")
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
            

            # Encontrar el píxel máximo (pico central)
            max_pos = np.unravel_index(np.argmax(autocorr_norm), autocorr_norm.shape)
            y0, x0 = max_pos
            max_val = autocorr_norm[y0, x0]

            # Mínimo de la imagen
            min_val = autocorr_norm.min()

            # Valor medio entre máximo y mínimo
            mid_val = (max_val + min_val) / 2

            # Crear malla de distancias desde el centro del pico
            Y, X = np.indices(autocorr_norm.shape)
            R = np.sqrt((X - x0)**2 + (Y - y0)**2)
            R = R.astype(int)

            # Perfil radial promedio
            r_max = R.max()
            radial_profile = np.array([autocorr_norm[R == r].mean() for r in range(r_max)])

            # Buscar el radio donde cae por debajo del valor medio (puedes ajustar criterio)
            r_mid = np.argmax(radial_profile < mid_val)

            self.pixel_medio = r_mid

            print(f"La intensidad cae al valor medio entre máximo y mínimo en un radio de {r_mid} píxeles.")


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


    ###    ###
    def cal_contraste(self):
        # Calcula el contraste de la imagen
        self.constraste = self.desviacion / self.media
        logging.info(f"Se ejecuta el método cal_contraste")
        logging.info(f"Valor contraste de la imagen {self.constraste}")
        return self.constraste
    
    
    ###   ###
    def cal_desviacion(self):
        # Calcula la desviación estándar de la imagen
        self.desviacion = np.std(self.spec)
        logging.info(f"Se ejecuta el método cal_desviacion")
        logging.info(f"Valor desviación estándar de la imagen {self.desviacion}")
        return self.desviacion
    
    
    #### Calcula la media de la imagen ###
    def cal_media(self):
        # Calcula la media de la imagen
        self.media = np.mean(self.spec)
        logging.info(f"Se ejecuta el método cal_media")
        logging.info(f"Valor promedio de la imagen = {self.media}")
        return self.media
    
    
    ###    ###    
    def cal_varianza(self):
        # Calcula la varianza de la imagen
        self.varianza = np.var(self.spec)
        logging.info(f"Se ejecuta el método cal_varianza")
        logging.info(f"Valor varianza de la imagen {self.varianza}")
        return self.varianza
    
    def filtro(self, show=True):
        # Aplica un filtro a la imagen y muestra el resultado
        if self.spec is None:
            raise ValueError("La imagen no ha sido cargada. Por favor, carga una imagen primero.")
        # Aplicar filtro de promediado de 3x3
        smoothed = uniform_filter(self.spec, size=8)

        if self.pixel_medio is None:
            self.autocorrelacion(imagen='o', show=False)

        # Tomar solo los píxeles cada 4 pasos para reducir
        self.filtrada = smoothed[1::self.pixel_medio, 1::self.pixel_medio]
        if show == True:
            # Mostrar la imagen original y la imagen filtrada
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(self.spec, cmap='gray')
            plt.title('Imagen original')
            
            plt.subplot(1, 2, 2)
            plt.imshow(self.filtrada, cmap='gray')
        
            plt.title('Imagen filtrada')
            plt.show()
        return self.filtrada                                                    
        logging.info(f"Se ejecuta el método filtro")


    #### Histograma de la imagen ###
    def histograma(self, imagen='o', bins_ = 256):
        # Genera un histograma de la imagen deseada
        c = bins_
        if imagen == 'o':
            print("Calculando el histograma de la imagen original...")
            image = self.spec.flatten()
            mu = self.media
        if imagen == 'm':
            print("Calculando el histograma de la imagen modificada...")
            image = self.imagef[~np.isnan(self.imagef)].flatten()
            mu = self.mediaf
        if imagen == 'f':
            print("Calculando el histograma de la imagen filtrada...")
            image = self.filtrada.flatten()
            mu = image.mean()
        # Crear histograma normalizado (como una densidad de probabilidad)
        
        a,b = np.histogram(image, bins = c, density=True)
        
        x = np.linspace(0,b[-1], 100)
        plt.hist(image, bins = c, density=True, label = 'Histograma de los datos experimentales')
        plt.plot(x, expon(scale = mu).pdf(x), label = 'curva teórica')

        plt.title('Histograma de la imagen')
        plt.xlabel('Niveles de gris')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.show()
        logging.info(f"Se ejecuta el método histograma")
    
    ### Presentación en pantalla de la imagen ###
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
            self.media = np.mean(self.spec)
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


    def miniprueba(self, imagen ='o'):

        if imagen == 'o':
            datos = self.spec
            mu = self.media
            print("Calculando la prueba de bondad de ajuste para la imagen original...")
        if imagen == 'f':
            if self.filtrada is None:
                self.filtro(show = False)
            print("Calculando la prueba de bondad de ajuste para la imagen filtrada...")    
            datos = self.filtrada
            mu = np.mean(self.filtrada)
        if imagen == 'm':
            if self.imagef is None:
                self.modificar(show = False)
            datos = self.imagef
            datos = datos[~np.isnan(datos)]
            mu = self.mediaf
            print("Calculando la pueba de bondad de ajuste para la imagen modificada...")
        
        def chi_sqr(dathistexp,dathistaj):
            return sum(((dathistexp-dathistaj)**2)/(dathistaj))
        valores, frecuencias = np.unique(datos[~np.isnan(datos)], return_counts=True)
        
        
        
        # chi=chi_sqr(frecuencias, np.sum(frecuencias)*expon(scale=mu).pdf(valores) )
        # print(chi)
        

        # df = len(valores)-2
        # alpha = 0.05
        # plt.bar(valores, frecuencias, width=np.ones(len(valores)), edgecolor='black',label='Observado')
        # plt.bar(valores, np.sum(frecuencias)*expon(scale=mu).pdf(valores), width=np.ones(len(valores)), edgecolor='red',alpha=0.5,label='Esperado')
        # plt.xlabel("Nivel de gris")
        # plt.ylabel("Frecuencia")
        # plt.title("Histogramas para la prueba de bondad")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        
        # chi2_critico = chi2.ppf(1-alpha, df)
        # print(f"Valor crítico de chi2 para {df} grados de libertad y alpha={alpha}: {chi2_critico:.2f}")
        # if chi < chi2_critico:
        #     print("No se rechaza la hipótesis nula: no hay nada que, estadísticamente, me diga que no se sigue una distribución exponencial negativa.")
        # else:
        #     print("Se rechaza la hipótesis nula: estadísticamente puedo decir que los datos no sigue una distribución exponencial negativa.")


        max_bins = 50
        bins = max_bins

        alpha = 0.05
        while bins > 1:
            counts, bin_edges = np.histogram(datos, bins=bins)
            if all(counts >= 5):
                break
            bins -= 1
        observed, _ = np.histogram(datos, bins=bin_edges)
        #observed = observed / np.sum(observed)  # Normalizar
        cdf_values = expon.cdf(bin_edges, scale=mu)
        expected = np.sum(observed)*np.diff(cdf_values)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        df = bins-2

        plt.bar(bin_centers, observed, width=np.diff(bin_edges), edgecolor='black',label='Observado')
        plt.bar(bin_centers, expected, width=np.diff(bin_edges), edgecolor='red',alpha=0.5,label='Esperado')
        plt.xlabel("Nivel de gris")
        plt.ylabel("Frecuencia")
        plt.title("Histogramas para la prueba de bondad")
        plt.grid(True)
        plt.legend()
        plt.show()
        chi=chi_sqr(observed, expected)
        print(chi)



        chi2_critico = chi2.ppf(1-alpha, df)
        print(f"Valor crítico de chi2 para {df} grados de libertad y alpha={alpha}: {chi2_critico:.2f}")
        if chi < chi2_critico:
            print("No se rechaza la hipótesis nula: no hay nada que, estadísticamente, me diga que no se sigue una distribución exponencial negativa.")
        else:
            print("Se rechaza la hipótesis nula: estadísticamente puedo decir que los datos no sigue una distribución exponencial negativa.")
    
    ###    ###
    def modificar(self, show=True):
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

        if show:
            # Grafica de ambas
            plt.plot(x, pdf_expon, 'r-', label='Distribución exponencial teórica')
            plt.title('Comparación con distribución de speckle completamente polarizado')
            plt.xlabel('Intensidad (ajustada)')
            plt.ylabel('Densidad de probabilidad')
            plt.legend()
            plt.grid(True)
            plt.show()
    
            
    ### Normalización de la imagen ###
    def normalizar(self):
        # Normaliza los  valores de intensidad de la imagen y actualiza
        # sus medeidas estadísticas.
        self.spec = self.spec / np.max(self.spec)
        logging.info(f"Se ejecuta el método normalizar")
        
    
    ###   ###    
    def prueba(self, imagen = 'o'):
        # Método para verificar que los datos son de un speckle completamente desarrollado, polarizado.
        if imagen == 'o':
            print("Calculando la prueba de hipótesis de ajuste para la imagen original...")
            mu = self.spec.flatten().mean()
            image = self.spec.flatten()
        if imagen == 'm':
            print("Calculando la prueba de hipótesis de ajuste para la imagen modificada...")
            if self.imagef is None:
                raise ValueError("La imagen no ha sido modificada. Por favor, modifica la imagen primero.")
            mu = self.mediaf
            # Filtrar valores NaNs
            image = self.imagef[~np.isnan(self.imagef)].flatten()
        if imagen == 'f':
            print("Calculando la prueba de hipótesis de ajuste para la imagen filtrada...")
            mu = self.filtrada.flatten().mean()
            image = self.filtrada.flatten()
        
        y, x = np.histogram(image, bins=256, range=None, density=True, weights=None)
        x = x[:-1]
        
        p_x = y # Densidad de probabilidad de X
        mask = p_x != 0 # Esta mascara nos permite quedarnos únicamente con los datos para los que poseemos información
        X_0 = np.copy(x)[mask]
        Y_0 = np.log(p_x[mask])
        
        Y = np.log(p_x) #Variable lineal

        #Estadigrafos de nuestra regresión

        b = (sum(X_0*Y_0)-len(X_0)*Y_0.mean()*X_0.mean())/(sum(X_0**2)-len(X_0)*X_0.mean()**2)
        a = Y_0.mean()-b*X_0.mean()

        plt.scatter(X_0,Y_0, label = 'dispersión', s=5)
        plt.plot(X_0,a+b*X_0, color = 'red', label = 'regresión')

        plt.title('Diagrama de dispersión de los datos')
        plt.xlabel('Intensidad del pixel')
        plt.ylabel('Log(Probabilidad)')
        plt.legend()
        plt.show()
        media_I = np.mean(X_0)
        media_LogP = np.mean(Y_0)
        Var_X = np.sum((X_0 - media_I)**2)/(len(X_0)-1)
        Var_Y = np.sum((Y_0 - media_LogP)**2)/(len(X_0)-1)
        s_X = np.sqrt(Var_X)
        
        S_YX =  np.sqrt((len(X_0)-1)/(len(X_0)-2) * (Var_Y - b**2 * Var_X))

        t = (b+1/mu)/(S_YX/(s_X*np.sqrt(len(X_0)-1)))


        # Hacemos la regresión lineal
        slope, intercept, r_value, p_value, std_err = linregress(X_0, Y_0)




        # Valor teórico de la pendiente si fuera exponencial
        b0 = -1 / mu
        print(f"Valor teórico de la pendiente: {b0}")
        print(f"Valor de la pendiente: {slope}")

        # Estadístico t para probar H0: b == b0
        t = (slope - b0) / std_err
        gl = len(X_0) - 2

        # Intervalo de aceptación al 90%
        intervalo = t_dist.interval(0.90, df=gl, loc=0, scale=1)

        print(f"Valor del estadístico t: {t}")
        print(f"Intervalo de aceptación (al 90%): {intervalo}")

        # Conclusión
        if t < intervalo[0] or t > intervalo[1]:
            print("Rechazamos H₀: la pendiente difiere de -1/μ")
        else:
            print("No se puede rechazar H₀: la pendiente podría ser -1/μ")

    ### Prueba de bondad con Chi2 ###
    def pruebaBondad(self, imagen='o'):
        if imagen == 'o':
            print("Calculando la prueba de bondad de ajuste para la imagen original...")
            mu = self.spec.flatten().mean()
            image = self.spec.flatten()
        if imagen == 'm':
            print("Calculando la prueba de bondad de ajuste para la imagen modificada...")
            mu = self.mediaf
            # Filtrar valores NaNs
            image = self.imagef[~np.isnan(self.imagef)].flatten()
        if imagen == 'f':
            print("Calculando la prueba de bondad de ajuste para la imagen filtrada...")
            mu = self.filtrada.flatten().mean()
            image = self.filtrada.flatten()

        max_bins = 50
        bins = max_bins

        while bins > 1:
            counts, bin_edges = np.histogram(image, bins=bins)
            if all(counts >= 5):
                break
            bins -= 1

        # Paso 2: calcular observados y esperados
        observed, _ = np.histogram(image, bins=bin_edges)
        

        # Calcular frecuencias esperadas usando la distribución exponencial negativa acumulada
        cdf_values = expon.cdf(bin_edges, scale=mu)
        expected = np.diff(cdf_values) * len(image)
        
        # Normalizar esperados
        #expected = expected * (observed.sum() / expected.sum())

        # Calcular las posiciones centrales de los bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Graficar
        plt.bar(bin_centers, observed, width=np.diff(bin_edges), edgecolor='black',label='Observado')
        plt.bar(bin_centers, expected, width=np.diff(bin_edges), edgecolor='red',alpha=0.5,label='Esperado')
        plt.xlabel("Nivel de gris")
        plt.ylabel("Frecuencia")
        plt.title("Histogramas para la prueba de bondad")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Paso 3: aplicar prueba chi-cuadrado
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected, ddof=1 , sum_check=False)

        print('media = {:.2f}'.format(mu))
        print('El valor de chi2 es {:.2f} y el valor p es {:.1f} %. El número de intervalos es {}'.format(chi2_stat, 100*p_value,bins))

    ### Comparación de histogramas por similitud de coseno ###
    def histograma_similaridad(self, imagen='o', bins_=256):
        # Genera un histograma de la imagen deseada y calcula la similitud de coseno
        c = bins_
        if imagen == 'o':
            print("Calculando el histograma de la imagen original...")
            image = self.spec.flatten()
            mu = self.media
        if imagen == 'm':
            print("Calculando el histograma de la imagen modificada...")
            image = self.imagef[~np.isnan(self.imagef)].flatten()
            mu = self.mediaf
        if imagen == 'f':
            print("Calculando el histograma de la imagen filtrada...")
            image = self.filtrada.flatten()
            mu = image.mean()

        # Crear histograma normalizado (como una densidad de probabilidad)
        # Paso 2: calcular observados y esperados
        observed, bin_edges = np.histogram(image, bins=c)

        # Calcular frecuencias esperadas usando la distribución exponencial negativa acumulada
        cdf_values = expon.cdf(bin_edges, scale=mu)
        expected = np.diff(cdf_values) * len(image)

        # Calcular las posiciones centrales de los bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calcular la similitud de coseno
        cos_sim = np.dot(observed, expected) / (np.linalg.norm(observed) * np.linalg.norm(expected))
        print(f"Similitud de coseno entre los histogramas: {cos_sim:.4f}")

        # Graficar
        plt.bar(bin_centers, observed, width=np.diff(bin_edges), edgecolor='black',label='Observado')
        plt.bar(bin_centers, expected, width=np.diff(bin_edges), edgecolor='red',alpha=0.5,label='Esperado')
        plt.xlabel("Nivel de gris")
        plt.ylabel("Frecuencia")
        plt.title("Histogramas para la prueba de similitud coseno")
        plt.text(0.05, 0.95, f'Similitud de coseno: {cos_sim:.4f}', transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        plt.grid(True)
        plt.legend()
        plt.show()

        logging.info(f"Se ejecuta el método histograma_similaridad")
        return cos_sim

    ### Cálculo de parámetros estadísticos ###
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
