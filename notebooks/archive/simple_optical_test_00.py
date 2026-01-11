from roman_disperser.optical_model import RomanOpticalModel
opt_model = RomanOpticalModel("data/Roman_grism_OpticalModel_v0.8.yaml")
opt_model.plot_quick_look()
opt_model.plot_quick_look(order="0")
opt_model.plot_quick_look(order="1")