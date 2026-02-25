Here are some notes on how to construct the axe trace file

```python
# Start with the SCA pixels
xc = 2043
yc = 2043
```

```python
# Rotate, since aXe disperses along the x-direction 
theta =  np.pi / 2

x_rot = int((np.cos(theta) * (xc-2044) - np.sin(theta) * (yc-2044)) + 2044)
y_rot = int((np.sin(theta) * (xc-2044) + np.cos(theta) * (yc-2044)) + 2044)
```

```python
# Set up the aXe configuration
# Load the code -- this is in the same directory, since we are just
# using it for tests
import grism_dispersion
grizli_conf = grism_dispersion.aXeConf(conf_file="./TestBuild_rot_det1.conf") 
dx = np.flip(np.arange(-380, 624, 1))
beam = grizli_conf.get_beam_trace(x=x_rot, y=y_rot, dx=dx)

# Get the beam in the original, un-rotated state
axe_beam_x = xc + beam[0]
axe_beam_y = yc - dx

# wavelengths in angstroms
axe_lam = beam[1]
```

We can now compare this with the outputs from our optical model. Let us restrict to 
wavelengths between 1 and 1.95 um, so do a selection on the `axe_*` arrays.


