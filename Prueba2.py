import PropsSpeckle as ps
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import j1
import pandas as pd
from scipy.signal import fftconvolve
from numpy.fft import fft2, ifft2, fftshift


espacio2 = ps.PropsSpeckle()
espacio2.imagen('4-150-1.tif',show=False)
espacio2.statisticspro()
acorr2 = espacio2.autocorrelacion(show=False)
pixel_size = 5.2  # Tamaño del píxel en micrómetros
H, W = espacio2.shape  



espacio2.modificar()



# acorrf1 = espacio2.autocorrelacionf()




# ## COMPARACIÓN DE LA AUTOCORRELACIÓN, ME FALTÓ EL PARÁMETRO Z CON LA FUNCIÓN DE BESSSEL ###
# pixel_size=5.2
# # Parámetros físicos
# D = 5000      # Diámetro de la apertura (en um)
# λ = 0.532    # Longitud de onda (en um, 500 nm)
# z = 180000      # Distancia al plano de observación (en um)
# I_bar = espacio2.mediaf   # Intensidad promedio

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


# H1, W1 = espacio2.shape  # Dimensiones de la imagen

# y1 = acorrf1[H1//2,:]

# x1 = np.linspace(-W1//2, W1//2, W1) * pixel_size  # En micrómetros
# plt.plot(x1, y1, c='blue')
# #plt.scatter(x, y, s=8, c='red')



# plt.plot(x, y_profile, c='black')
# #plt.scatter(x, y_profile, s=8, c='red')
# plt.title('Autocorrelación')
# plt.xlabel('Desplazamiento en x (µm)')
# plt.ylabel('Autocorrelación')
# plt.show()


espacio2.prueba()