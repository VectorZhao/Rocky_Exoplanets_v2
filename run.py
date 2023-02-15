from deepexo.rockyplanet import RockyPlanet

M = 1.77  # mass in Earth masses
R = 1.228  # radius in Earth radii
cFeMg = 0.685  # bulk Fe/(Mg + Si) (molar ratio)
k2 = 0.819  # tide Love number

planet_params = [
    M,
    R,
    cFeMg,
    k2,
]
rp = RockyPlanet()
pred = rp.predict(planet_params)
rp.plot(pred)
# model_a = rp.load_model("model/model_a.h5")
# model_b = rp.load_model("model/model_b.h5")
# model_a_scaler = rp.load_scaler("model/model_a_scaler.save")
# model_b_scaler = rp.load_scaler("model/model_b_scaler.save")

# pred = rp.predict(planet_params, model_a, model_a_scaler)