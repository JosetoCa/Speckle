# -*- coding: utf-8 -*-
"""

"""
import numpy as np


class PropsSpeckle:

    # Class attribute
    version = '0.0.2'
    name = 'Plane objects'
    units = 'Millimeters'

    # Instance attributes
    def __init__(self, L = 10.0, N = 1024):
        """
        Parameters
        ----------
        L : float, optional
            Side length of working area. The default is 10.0 mm.
        N : integer, optional
            Side length of array. The default is 1024.

        Returns
        -------
        None.

        """
        self.L = L
        self.N = N