The following is a plan to extend my star dispersion routines to galaxies. 

*Input* : A galaxy image specified on a 2D grid, galaxy spectrum on a 1D grid, undispersed position (X,Y) on the detector.
1. Compared to the star case, this just differs in that we are also passing in an image. 

*Constraints/Assumptions* 
1. Assume that the galaxy is small enough that there is no significant flux near the boundaries, either before/after PSF convolution. 
2. Assume that the oversampling is the same as what we have for the PSF.

*Algorithm*

1. Calculate the dispersed galaxy shape :
	1. Start with the galaxy reproduced at the same wavelengths as the PSF grids.
	2. Calculate the dispersed position at the center of these grids.
	3. Calculate the relative dispersed position for the points on the grid. We could do this either by computing the Jacobian of the dispersion solution and then just doing a simple linear approximation, or by just running the dispersion code for each of the points. We need to decide which of these is better.
	4. Use a simple bilinear interpolation to put the flux at the relative dispersed positions back onto the grid. Drop points that fall off the grid.
2. Convolve with the PSF :
	1. Now, since the galaxies are at the same wavelengths as the PSF grid, convolve these dispersed galaxies with the PSF grid. We will probably need to shift the PSF images, since the center of the PSF is at the center of the grid. Do this with the appropriate zero padding. Probably best to do the convolution with an FFT.
3. Now we should have a structure that is close to identical to our star case. At this point, we proceed as we did for the stars
	1. Loop over the wavelength grid.
	2. At each wavelength, interpolate the galaxy image from step 2
	3. Deposit a scaled version of these images based on the spectrum using a simple nearest grid point scheme. 

Some notes on step 3 : we might follow the `star_disperser.py` routine and batch the wavelengths. 
