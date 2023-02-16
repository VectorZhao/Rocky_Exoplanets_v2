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
