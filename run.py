from deepexo.rockyplanet import RockyPlanet

M = 1.77,  # mass in Earth masses
R = 1.228,  # radius in Earth radii
cFeMg = 0.685,  # bulk Fe/(Mg + Si) (molar ratio)
k2 = 0.819, # tide Love number

planet_params = [M, R, cFeMg, k2]
rp = RockyPlanet()
pred = rp.predict(planet_params)