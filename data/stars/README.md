This is the data directory in the star_fields repository here :
   https://github.com/roman-grs-pit/star_fields/tree/main

copied here for convenience, since this is a small data set. 

The conversion from the star_template_index to the appropriate line in 
SEDtemplates/input_spectral_STARS.lis is:

temp_inds = stars['star_template_index'] - 58*(stars['star_template_index']//58)

The input_spectral_STARS.lis file contains filenames to spectra in the same directory.
Lines with "garbage" are unused slots (these should never be referenced in the file).

