from tensorflow.keras.models import load_model
import numpy as np

import mdn
import joblib
from numpy.typing import ArrayLike
class RockyPlanet:
    def predict(self, planet_params: ArrayLike):
        """
        Args:
            Array of floats of the planet parameters
            in the following order: [M, R, k2].
            The following input ranges are supported:
                M: 0.1 < M [M_earth] < 30
                R: 0.1 < R [R_earth] < 30
                k2: 0 < k2 < 0.5
        Returns:

        """

