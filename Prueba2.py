import PropsSpeckle as ps
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import j1
import pandas as pd
from scipy.signal import fftconvolve
from numpy.fft import fft2, ifft2, fftshift

espacio1 = ps.PropsSpeckle()

#Se llama una imagen, en este caso se da el nombre
espacio1.imagen('3-16.tif',show=False)
espacio1.statisticspro()
# espacio.histograma()
acorr1 = espacio1.autocorrelacion(show=False)

espacio2 = ps.PropsSpeckle()
espacio2.imagen('4-150.tif',show=False)
espacio2.statisticspro()
acorr2 = espacio2.autocorrelacion(show=False)
pixel_size = 5.2  # Tamaño del píxel en micrómetros
H, W = espacio1.shape  

x = np.linspace(-W//2, W//2, W) * pixel_size
y1 = acorr1[H//2,:]
y1 = y1/(espacio1.media**2)  
y2 = acorr2[H//2,:]
y2 = y2/(espacio2.media**2)




plt.plot(x, y1, c='blue', label='3-16.tif')
plt.plot(x, y2, c='red', label='4-150.tif')
plt.title('Autocorrelación')
plt.xlabel('Desplazamiento en x (µm)')
plt.ylabel('Autocorrelación')
plt.legend()
plt.show()


espacio1.modificar()
espacio2.modificar()


# Supongamos que ya cargaste la imagen y se llama espacio1.imagef
imagen1 = espacio1.imagef
imagen_temp1 = np.nan_to_num(imagen1, nan=0.0)
mask1 = ~np.isnan(imagen1)
mask1 = mask1.astype(float)

# FFT de la imagen y la máscara
fft_img1 = fft2(imagen_temp1)
fft_mask1 = fft2(mask1)

# Autocorrelación de la imagen y de la máscara
autocorr_img1 = fftshift(np.real(ifft2(fft_img1 * np.conj(fft_img1))))
autocorr_mask1 = fftshift(np.real(ifft2(fft_mask1 * np.conj(fft_mask1))))

# Normalizar (evitar división por cero)
autocorr_norm1 = np.divide(
    autocorr_img1,
    autocorr_mask1,
    out=np.zeros_like(autocorr_img1),
    where=autocorr_mask1 > 0
)

# Normalizar a 1 el máximo para mejor visualización


# Mostrar
y = autocorr_norm1[H//2,:]
# y = y/(espacio1.mediaf**2)  # Normalizar por la media al cuadrado
plt.plot(x, y, c='blue', label='3-16.tif')





# Supongamos que ya cargaste la imagen y se llama espacio1.imagef
imagen2 = espacio2.imagef
imagen_temp2 = np.nan_to_num(imagen2, nan=0.0)
mask2 = ~np.isnan(imagen2)
mask2 = mask2.astype(float)

# FFT de la imagen y la máscara
fft_img2 = fft2(imagen_temp2)
fft_mask2 = fft2(mask2)

# Autocorrelación de la imagen y de la máscara
autocorr_img2 = fftshift(np.real(ifft2(fft_img2 * np.conj(fft_img2))))
autocorr_mask2 = fftshift(np.real(ifft2(fft_mask2 * np.conj(fft_mask2))))

# Normalizar (evitar división por cero)
autocorr_norm2 = np.divide(
    autocorr_img2,
    autocorr_mask2,
    out=np.zeros_like(autocorr_img2),
    where=autocorr_mask2 > 0
)


y = autocorr_norm2[H//2,:]
# y = y/(espacio2.mediaf**2)  # Normalizar por la media al cuadrado
plt.plot(x, y, c='red', label='4-150.tif')
plt.legend()
plt.title('Autocorrelación')
plt.xlabel('Desplazamiento en x (µm)')
plt.ylabel('Autocorrelación')
plt.show()
espacio1.autocorrelacionf()




### COMPARACIÓN DE LA AUTOCORRELACIÓN, ME FALTÓ EL PARÁMETRO Z CON LA FUNCIÓN DE BESSSEL ###
# pixel_size=5.2
# # Parámetros físicos
# D = 5000       # Diámetro de la apertura (en um)
# λ = 0.532    # Longitud de onda (en um, 500 nm)
# z = 1000000      # Distancia al plano de observación (en um)
# I_bar = espacio.media   # Intensidad promedio

# # Definimos el plano (x, y)
# H = pixel_size*512  # tamaño del plano en um
# L = pixel_size*640
# res = 4000
# x = np.linspace(-L, L, res)
# y = np.linspace(-H, H, res)
# X, Y = np.meshgrid(x, y)
# r = np.sqrt(X**2 + Y**2)

# # Argumento del Bessel
# beta = (np.pi * D * r) / (λ * z)
# # Evitamos división por cero con np.where
# J1_term = np.where(beta != 0, (2 * j1(beta) / beta), 1.0)

# # Función Γ_I(r)
# Gamma_I = I_bar**2 * (1 + np.abs(J1_term)**2)

# # Gráfica

# center = res // 2
# y_profile = Gamma_I[center, :]
# y_profile = y_profile - 268.5  # Se le resta el valor de la media
# y_profile = y_profile/np.max(y_profile)


# H1, W1 = espacio.shape  # Dimensiones de la imagen

# y1 = acorr[H1//2,:]
# y1 = y1 -3.56e8  # Se le resta el valor de la media
# y1 = y1/np.max(y1)  # Se normaliza la autocorrelación

# x1 = np.linspace(-W1//2, W1//2, W1) * pixel_size  # En micrómetros
# plt.plot(x1, y1, c='blue')
# #plt.scatter(x, y, s=8, c='red')



# plt.plot(x, y_profile, c='black')
# #plt.scatter(x, y_profile, s=8, c='red')
# plt.title('Autocorrelación')
# plt.xlabel('Desplazamiento en x (µm)')
# plt.ylabel('Autocorrelación')
# plt.show()