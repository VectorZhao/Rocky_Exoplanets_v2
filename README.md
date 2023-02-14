## Overview
🤖Machine learning (ML) promises dramatic speedups in characterizing the interior structure of exoplanets. Among various ML techniques, mixture density networks (MDNs) are quite appropriate to degenerate cases without sacrificing degenerate solutions. The MDN model requires large sets of data for training and yields multimodal probability distributions for various target variables through inputs of known quantities. Such a data-driven approach decouples interior model
calculations from ML inferences and hence a well-trained ML model is capable of quickly characterizing planetary interiors. [Baumeister et al. (2020)](https://iopscience.iop.org/article/10.3847/1538-4357/ab5d32) applied MDN-based ML to infer the distribution of possible thicknesses of each planetary layer for exoplanets up to 25 Earth masses, where MDN inference for one planet takes only few miliseconds compared with the inversion computing time of potentially several hours. In [Zhao & Ni (2021)](https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html) and [Zhao & Ni (2022)](https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html), the MDN was used to simultaneously predict the layer thicknesses and core properties of exoplanets including Earth-like planets and gas giants.

In this work, We trained a machine learning model by using MDN algorithm to quickly and eﬃciently infer the interior structure of rocky exoplanets with large compositional diversity.

## Quick Start
### Step 1:
[Fork and clone](https://help.github.com/articles/fork-a-repo) a copy of the `Rocky_Exoplanets_v2` repository to your local machine.

### Step 2:
Download [`Anaconda`](https://www.anaconda.com/products/individual#Downloads) and install it on your machine.
Create a `conda` environment called `Rocky_Exoplanets_v2` and install all the necessary dependencies:

    $ conda create -n Rocky_Exoplanets pip python=3.7.6 keras-mdn-layer jupyter
    
### Step 3:
Activate the `Rocky_Exoplanets_v2` environment:

    $ conda activate Rocky_Exoplanets_v2

### Step 4:
Change into your local copy of the `Rocky_Exoplanets_v2` repo:

    $ cd /you own path/Rocky_Exoplanets_v2

### Step 5:
Install the requirments for predicing in the current Conda environment:

    $ pip install -r requirements.txt

### Step 6:
Open `Jupyter Notebook` and load the file `MDN_two_planets_prediction.ipynb`:

    $ jupyter notebook

At this point you are ready to start investigating the interiors of rocky exoplanets!
## References
[https://iopscience.iop.org/article/10.3847/1538-4357/ab5d32](https://iopscience.iop.org/article/10.3847/1538-4357/ab5d32)

[https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html](https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html)

[https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html](https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html)
