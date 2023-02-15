from tensorflow.keras.models import load_model
import numpy as np

import mdn
import joblib

OUTPUT_DIMS = 6  # 6 outputs:
# 'H2O_radial_frac', 'Mantle_radial_frac', 'Core_radial_frac', 'Core_mass_frac', 'P_CMB', 'T_CMB',
N_MIXES = 20  # 20 mixtures


class RockyPlanet:
    def __init__(self):
        self.model_a = load_model(r"model/model_a.h5", custom_objects={
            'MDN': mdn.MDN(OUTPUT_DIMS, N_MIXES),
            "mdn_loss_func": mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)}, compile=False)
        self.model_a_scaler = joblib.load(r"model/model_a_scaler.save")
        self.model_b = load_model(r"model/model_b.h5", custom_objects={
            'MDN': mdn.MDN(OUTPUT_DIMS, N_MIXES), "mdn_loss_func": mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)})
        self.model_b_scaler = joblib.load(r"model/model_b_scaler.save")

    def predict(self, planet_params):
        if len(planet_params) == 3:
            print("正在调用inputs={}的model A进行预测".format(planet_params))
            scaled_params = self.model_a_scaler.transform(np.array(planet_params).reshape(1, -1))
            return self.model_a.predict(scaled_params)[0]
        elif len(planet_params) == 4:
            print("正在调用inputs={}的model B进行预测".format(planet_params))
            scaled_params = seyhlf.model_b_scaler.transform(np.array(planet_params).reshape(1, -1))
            return self.model_b.predict(scaled_params)[0]
        else:
            raise ValueError(
                "Invalid number of planet parameters. Expected 3 or 4, but got {}".format(len(planet_params)))
