## Overview

![MDN_Cartoon](https://user-images.githubusercontent.com/16644993/196080290-b73adcc4-c65b-4a57-91b9-5dc08a903d00.jpg)

We trained a machine learning model by using mixture density network (MDN) algorithm to quickly and eﬃciently infer the interior structure of rocky exoplanets with large compositional diversity.

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
