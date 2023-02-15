from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import math
import mdn
import joblib
import os

OUTPUT_DIMS = 6  # 6 outputs:
# 'H2O_radial_frac', 'Mantle_radial_frac', 'Core_radial_frac', 'Core_mass_frac', 'P_CMB', 'T_CMB',
N_MIXES = 20  # 20 mixtures

# print(os.getcwd())
model_path = os.path.join(os.getcwd(), "deepexo/model")


class RockyPlanet:
    def __init__(self):
        # print('init RockyPlanet')
        self.model_a = load_model(os.path.join(model_path, "model_a.h5"), custom_objects={
            'MDN': mdn.MDN(OUTPUT_DIMS, N_MIXES),
            "mdn_loss_func": mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)}, compile=False)
        self.model_a_scaler = joblib.load(os.path.join(model_path, "model_a_scaler.save"))
        self.model_b = load_model(os.path.join(model_path, "model_b.h5"), custom_objects={
            'MDN': mdn.MDN(OUTPUT_DIMS, N_MIXES),
            "mdn_loss_func": mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)}, compile=False)
        self.model_b_scaler = joblib.load(os.path.join(model_path, "model_b_scaler.save"))

    def predict(self, planet_params):
        """Predictes the .
        if len(planet_params) == 3:
            print("正在调用inputs={}的model A进行预测".format(planet_params))
            scaled_params = self.model_a_scaler.transform(np.array(planet_params).reshape(1, -1))
            return self.model_a.predict(scaled_params)
        elif len(planet_params) == 4:
            print("正在调用inputs={}的model B进行预测".format(planet_params))
            scaled_params = self.model_b_scaler.transform(np.array(planet_params).reshape(1, -1))
            return self.model_b.predict(scaled_params)
        else:
            raise ValueError(
                "Invalid number of planet parameters. Expected 3 or 4, but got {}".format(len(planet_params)))

    def plot(self, pred):

        (y_min1, y_max1, y_min2, y_max2, y_min3, y_max3,
         y_min4, y_max4, y_min5, y_max5, y_min6, y_max6) = \
            0.00015137, 0.145835, 0.127618, 0.973427, 0.00787023, 0.799449, 1.17976e-06, 0.699986, 10.7182, 1999.49, 1689.37, 5673.87
        mus = np.apply_along_axis((lambda a: a[:N_MIXES * OUTPUT_DIMS]), 1, pred)
        sigs = np.apply_along_axis((lambda a: a[N_MIXES * OUTPUT_DIMS:2 * N_MIXES * OUTPUT_DIMS]), 1, pred)
        pis = np.apply_along_axis((lambda a: mdn.softmax(a[-N_MIXES:])), 1, pred)

        for m in range(OUTPUT_DIMS):
            locals()['mus' + str(m)] = []
            locals()['sigs' + str(m)] = []
            for n in range(20):
                locals()['mus' + str(m)].append(mus[0][n * OUTPUT_DIMS + m])
                locals()['sigs' + str(m)].append(sigs[0][n * OUTPUT_DIMS + m])
        x_max = [1, 1, 1, 1, 1, 1]
        x_maxlabels = [
            y_min1 + (y_max1 - y_min1) * x_max[0],
            y_min2 + (y_max2 - y_min2) * x_max[1],
            y_min3 + (y_max3 - y_min3) * x_max[2],
            y_min4 + (y_max4 - y_min4) * x_max[3],
            y_min5 + (y_max5 - y_min5) * x_max[4],
            y_min6 + (y_max6 - y_min6) * x_max[5],
        ]

        wrf = []  # H2O_radial_frac
        mrf = []  # Mantle_radial_frac
        crf = []  # Core_radial_frac
        cmf = []  # Core mass frac
        pcmb = []  # CMB temperature
        tcmb = []  # CMB pressure

        for x1 in np.arange(0.02, 0.15, 0.04):
            wrf.append((x1 - y_min1) / (x_maxlabels[0] - y_min1) * x_max[0])
        for x2 in np.arange(0.2, 1, 0.2):
            mrf.append((x2 - y_min2) / (x_maxlabels[1] - y_min2) * x_max[1])
        for x3 in np.arange(0.1, 0.9, 0.2):
            crf.append((x3 - y_min3) / (x_maxlabels[2] - y_min3) * x_max[2])
        for x4 in np.arange(0.1, 0.8, 0.2):
            cmf.append((x4 - y_min4) / (x_maxlabels[3] - y_min4) * x_max[3])
        for x5 in np.arange(200, 2000, 400):
            pcmb.append((x5 - y_min5) / (x_maxlabels[4] - y_min5) * x_max[4])
        for x6 in np.arange(2000, 6000, 1000):
            tcmb.append((x6 - y_min6) / (x_maxlabels[5] - y_min6) * x_max[5])

        xticklabels = [[round(x, 2) for x in np.arange(0.02, 0.15, 0.04)],
                       [round(x, 2) for x in np.arange(0.2, 1, 0.2)],
                       [round(x, 2) for x in np.arange(0.1, 0.9, 0.2)],
                       [round(x, 2) for x in np.arange(0.1, 0.8, 0.2)],
                       [round(x, 2) for x in np.arange(200, 2000, 400)],
                       [round(x, 2) for x in np.arange(2000, 6000, 1000)],
                       ]
        xticks = [wrf, mrf, crf, cmf, pcmb, tcmb]

        colors = [
            "steelblue",
            "#EA7D1A",
            "red",
            "gold",
            "#2ecc71",
            "#03a9f4"
        ]
        predict_label = [
            "Water radial fraction",
            "Mantle radial fraction",
            "Core radial fraction",
            "Core mass fraction",
            r"CMB pressure ($10^2$ GPa)",
            r"CMB temperature ($10^3$ K)"
        ]

        y_label = np.arange(0, 1, 0.001).reshape(-1, 1)
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(hspace=0.7, wspace=0.2)
        for i in range(OUTPUT_DIMS):
            ax = fig.add_subplot(3, 2, i + 1)
            mus_ = np.array(locals()['mus' + str(i)])
            sigs_ = np.array(locals()['sigs' + str(i)])
            factors = 1 / math.sqrt(2 * math.pi) / sigs_
            exponent = np.exp(-1 / 2 * np.square((y_label - mus_) / sigs_))
            GMM_PDF = np.sum(pis[0] * factors * exponent, axis=1)  # Summing multiple Gaussian distributions
            plt.plot(
                y_label,
                GMM_PDF,
                color=colors[i],
                #         label=label,
                lw=2,
                zorder=10,
            )
            ax.set_xlim(0, x_max[i])
            ax.set_ylim(bottom=0)

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            #     print(xticks[i])
            #     xticklabels[i]
            ax.set_xticks(xticks[i])
            ax.set_xticklabels(xticklabels[i])
            ax.set_xlabel(predict_label[i])

        return plt.show()
