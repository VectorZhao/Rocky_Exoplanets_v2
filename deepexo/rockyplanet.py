from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import math
import functools
import mdn
import joblib
import os

OUTPUT_DIMS = 6  # 6 outputs:
# 'H2O_radial_frac', 'Mantle_radial_frac', 'Core_radial_frac', 'Core_mass_frac', 'P_CMB', 'T_CMB',
N_MIXES = 20  # 20 mixtures

# print(os.getcwd())
model_path = os.path.join(os.getcwd(), "deepexo/model")

def lazy_property(fn):
    attr_name = "_lazy_" + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
class RockyPlanet:
    """A class for characterizing the interior structure of rocky exoplanets."""
    def __init__(self):
        pass

    @lazy_property
    def model_a(self):
        return load_model(os.path.join(model_path, "model_a.h5"), custom_objects={
            'MDN': mdn.MDN(OUTPUT_DIMS, N_MIXES),
            "mdn_loss_func": mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)}, compile=False)

    @lazy_property
    def model_a_scaler(self):
        return joblib.load(os.path.join(model_path, "model_a_scaler.save"))

    @lazy_property
    def model_b(self):
        return load_model(os.path.join(model_path, "model_b.h5"), custom_objects={
            'MDN': mdn.MDN(OUTPUT_DIMS, N_MIXES),
            "mdn_loss_func": mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)}, compile=False)

    @lazy_property
    def model_b_scaler(self):
        return joblib.load(os.path.join(model_path, "model_b_scaler.save"))
    def predict(self, planet_params: object) -> object:
        """Predicts the Water radial fraction, Mantle radial fraction, Core radial fraction, Core mass fraction,
        CMB pressure and CMB temperature for the given planetary parameters in terms of planetary mass M [M_Earth],
        radius [R_Earth], bulk Fe/(Mg + Si) (molar ratio), and tidal Love number k2.

        Args:
            planet_params (list): A list of planetary parameters in the order of [M, R, k2] or [M, R, cFeMg, k2].
        Returns:
            pred: contains parameters for distributions, not actual points on the graph.
        """
        if len(planet_params) == 3:
            print("Predicting using model A with inputs={}".format(planet_params))
            scaled_params = self.model_a_scaler.transform(np.array(planet_params).reshape(1, -1))
            return self.model_a.predict(scaled_params)
        elif len(planet_params) == 4:
            print("Predicting using model B with inputs={}".format(planet_params))
            scaled_params = self.model_b_scaler.transform(np.array(planet_params).reshape(1, -1))
            return self.model_b.predict(scaled_params)
        else:
            raise ValueError(
                "Invalid number of planet parameters. Expected 3 or 4, but got {}".format(len(planet_params)))

    def plot(self, pred: object, save: object = False, filename: object = "pred.png") -> object:
        """Plots the predicted distributions for Water radial fraction, Mantle radial fraction, Core radial fraction,
        Core mass fraction, CMB pressure and CMB temperature.
        Args:
            pred (array): contains parameters for distributions, not actual points on the graph.
            save (bool, optional): Defaults to False. If True, saves the plot to a file.
            filename (str, optional): Defaults to "". The filename to save the plot to.
        Returns:
            None or saves the plot to a file.
        """
        print("###############################################")
        print("Plotting...")
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
        x_max_labels = [
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
            wrf.append((x1 - y_min1) / (x_max_labels[0] - y_min1) * x_max[0])
        for x2 in np.arange(0.2, 1, 0.2):
            mrf.append((x2 - y_min2) / (x_max_labels[1] - y_min2) * x_max[1])
        for x3 in np.arange(0.1, 0.9, 0.2):
            crf.append((x3 - y_min3) / (x_max_labels[2] - y_min3) * x_max[2])
        for x4 in np.arange(0.1, 0.8, 0.2):
            cmf.append((x4 - y_min4) / (x_max_labels[3] - y_min4) * x_max[3])
        for x5 in np.arange(200, 2000, 400):
            pcmb.append((x5 - y_min5) / (x_max_labels[4] - y_min5) * x_max[4])
        for x6 in np.arange(2000, 6000, 1000):
            tcmb.append((x6 - y_min6) / (x_max_labels[5] - y_min6) * x_max[5])

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
        fig.subplots_adjust(hspace=0.7, wspace=0.4)
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
                lw=2,
                zorder=10,
            )
            ax.set_xlim(0, x_max[i])
            ax.set_ylim(bottom=0)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xticks(xticks[i])
            ax.set_xticklabels(xticklabels[i])
            ax.set_xlabel(predict_label[i])
            ax.set_ylabel("Probability density")
        if save:
            print("Saving figure to {}".format(filename))
            return plt.savefig(filename, dpi=300)
        else:
            print("Showing figure")
            return plt.show()
