The code aims to disperse the flux in a box in x-y-lambda into a region x'-y' following a coded optical model. The box is divided into a set of cells defined by a set of grid points. Currently the code calculates the dispersed position of the center of each cell, and then distributes the flux in the cell by a bilinear interpolation onto the dispersed grid. This isn't correct, since the flux could fall completely into one cell - the input cell might be a lot smaller.

The first part of the fix is to replace the bilinear interpolation with a simpler grid deposition.

To be more careful with how the flux gets distributed, one needs relatively smaller grid cells, and computing the full dispersion solution would get expensive for these points. So instead we propose the following algorithm :

1. Divide the x-y-lambda into cells; these could be "relatively large cells".
2. Calculate the dispersion solution for the central point, but also calculate the Jacobian of that transformation. This is easy since the code is all in JAX.
3. Allocate points using a Sobol sequence within the cell.
4. Calculate the flux at each point using the input image and spectrum.
5. Disperse the points using the Jacobian. Put these points in a smaller subgrid, also save the position of the subgrid.
6. Repeat 3-5 for all cells.
7. Normalize all the dispersed points by the full flux in the input box.
8. Deposit each of the subgrids into the full 4088x4088 grid.


## Implementation Plan

### Step 0

Let's test to see that the Jacobian calculation is accurate enough. Here's how I propose doing it. Let 
us consider cells that are (10 pix x 10 pix) x 100A where the first two are in SCA pixel units. 
Run this from -500 pix to 5500 pix and 0.9 um to 2um in wavelength. For eacj of these cells, take the 8 corners and see where they get displaced to, both using the full solution as well as the Jacobian, and 
measure the largest displacement in the output x',y'.

We should compute the full Jacobian, instead of relying on the Jacobian vector-product, since we're 
going to be running this against many points.

