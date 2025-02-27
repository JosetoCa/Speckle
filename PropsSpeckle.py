# -*- coding: utf-8 -*-
"""

"""
import numpy as np


class PropsSpeckle:

    # Class attribute
    version = '0.0.1.0'
    name = 'Caracterización de difusores'

    # Instance attributes
    def __init__(self, archivo="ruta.txt"):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                self.primera_linea = f.readline().strip()  # Leer la primera línea y quitar espacios en blanco
        except FileNotFoundError:
            print(f"Error: El archivo '{archivo}' no se encontró.")
            self.primera_linea = None
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")
            self.primera_linea = None
