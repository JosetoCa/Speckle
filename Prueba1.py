import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import linregress

# Carga imágenes en escala de grises
original = Image.open("C:\\Proyectos\\Speckle\\images\\prueba1.tif").convert("L")
corregida = Image.open("C:\\Proyectos\\Speckle\\images\\prueba1.8.tif").convert("L")
# Convierte a arrays numpy y normaliza al rango [0, 1]
I_orig = np.asarray(original, dtype=np.float32) / 255.0
I_corr = np.asarray(corregida, dtype=np.float32) / 255.0

# Aplana las imágenes para análisis 1D
x = I_orig.flatten()
y = I_corr.flatten()

# Filtra valores muy pequeños o cero (evita log(0))
epsilon = 1e-6
mask = (x > epsilon) & (y > epsilon)
x = x[mask]
y = y[mask]

# Aplica logaritmo
logx = np.log(x)
logy = np.log(y)

# Ajuste lineal log-log: log(y) = gamma * log(x)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)

# Resultados
print(f"Exponente estimado (gamma): {slope:.4f}")
print(f"R² del ajuste: {r_value**2:.4f}")

# Gráfica
plt.figure(figsize=(6, 6))
plt.scatter(logx, logy, s=0.5, alpha=0.3, label="Datos")
plt.plot(logx, slope * logx + intercept, color='red', label=f'Ajuste: y = {slope:.2f}x')
plt.xlabel('log(I_original)')
plt.ylabel('log(I_corregida)')
plt.title('Ajuste log-log para estimar exponente gamma')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

