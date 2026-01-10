#%%
import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
import os

#%%
# Get the pixi path and load the optical model
pixi_root_path=os.environ.get("PIXI_PROJECT_ROOT",".")
fn = os.path.join(pixi_root_path, "data/Roman_grism_OpticalModel_v0.8.yaml")
opt_model = RomanOpticalModel(fn)

# %%
sca = omj.make_sca_payload(opt_model, 1, "1")

# %%
