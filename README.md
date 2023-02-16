## Overview
🤖Machine learning (ML) promises dramatic speedups in characterizing the interior structure of exoplanets. Among various ML techniques, mixture density networks (MDNs) are quite appropriate to degenerate cases without sacrificing degenerate solutions. The MDN model requires large sets of data for training and yields multimodal probability distributions for various target variables through inputs of known quantities. Such a data-driven approach decouples interior model
calculations from ML inferences and hence a well-trained ML model is capable of quickly characterizing planetary interiors. [Baumeister et al. (2020)](https://iopscience.iop.org/article/10.3847/1538-4357/ab5d32) applied MDN-based ML to infer the distribution of possible thicknesses of each planetary layer for exoplanets up to 25 Earth masses, where MDN inference for one planet takes only few miliseconds compared with the inversion computing time of potentially several hours. In [Zhao & Ni (2021)](https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html) and [Zhao & Ni (2022)](https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html), the MDN was used to simultaneously predict the layer thicknesses and core properties of exoplanets including rocky planets with Earth-like composition and gas giants.

In this work, We trained a machine learning model by using MDN algorithm to quickly and eﬃciently infer the interior structure of rocky exoplanets with **large compositional diversity.**

## Machine Learning Models
We provided two machine learning models for uses: **Model A** trained on `[M, R, k2]` inputs and **Model B** trained on `[M, R, Fe/(Mg + Si), k2]` inputs. `M`, `R`, and `Fe/(Mg + Si)` represent the mass, radius, and bulk Fe/Si and Mg/Si abundance ratios of the planet, respectively, and `k2` is the tidal Love number. The relative refractory composition of rocky exoplanets can be well constrained by the elemental abundances of their host stars. In [Adibekyan et al. (2021)](https://www.science.org/doi/10.1126/science.abg8794), a strong correlation with a slope ∼5 was achieved between the Fe/(Mg+Si) abundance ratios of the planets and of their host stars, 

 $$(\frac{Fe}{Mg + Si})_{planet} = -1.35 \pm 0.36 + 4.84 \pm 0.92 \times (\frac{Fe}{Mg + Si})_{star}$$


**Model A** has a better predictive accuracy, but its application is limited by some difficulties in measuring the tidal Love number `k2` of rocky exoplanets. **Model B** significantly breaks the density-composition degeneracy and accurately predicts the interior properties of rocky exoplanets. Along with the development of space-based observation technologies, orbital or shape observations could be possible to determine the tidal Love number `k2` of rocky exoplanets and hence the machine learning models **B** and **C** would be applied more broadly.

## Quick Start
### Step 1:
[Fork and clone](https://help.github.com/articles/fork-a-repo) a copy of the `Rocky_Exoplanets_v2` repository to your local machine.

### Step 2:
Download [`Anaconda`](https://www.anaconda.com/products/individual#Downloads) and install it on your machine.
Create a `conda` environment called `Rocky_Exoplanets` and install all the necessary dependencies:

    $ conda create -n Rocky_Exoplanets pip python=3.7.6 jupyter
    
### Step 3:
Activate the `Rocky_Exoplanets` environment:

    $ conda activate Rocky_Exoplanets

### Step 4:
Change into your local copy of the `Rocky_Exoplanets_v2` repo:

    $ cd /you own path/Rocky_Exoplanets_v2

### Step 5:
Install the requirments for predicing in the current Conda environment:

    $ pip install -r requirements.txt

## Usage

```python
from deepexo.rockyplanet import RockyPlanet

# Keplar-78b
M = 1.77  # mass in Earth masses
R = 1.228  # radius in Earth radii
cFeMgSi = 0.685  # bulk Fe/(Mg + Si) (molar ratio)
k2 = 0.819  # tidal Love number

planet_params = [
    M,
    R,
    cFeMgSi,
    k2,
]
rp = RockyPlanet()
pred = rp.predict(planet_params)
rp.plot(pred,
        save=True,  # save to file
        filename="pred.png"  # save to current directory, filename is pred.png, you can change the extension to eps
        # or pdf
        )

```
## References
[https://www.science.org/doi/10.1126/science.abg8794](https://www.science.org/doi/10.1126/science.abg8794)

[https://iopscience.iop.org/article/10.3847/1538-4357/ab5d32](https://iopscience.iop.org/article/10.3847/1538-4357/ab5d32)

[https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html](https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html)

[https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html](https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html)
