"""
Utilities for Roman Optical Model
"""

import warnings
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle


class RomanDetectorCoordinates:
    """
    Helper class to handle all the coordinate transformations for WFI
    """

    def __init__(
        self,
        naxis1,
        naxis2,
        crpix1,
        crpix2,
        pos_angle_detector,
        pixel_scale,
        plate_scale,
        xy_centers,
    ):  # pylint: disable=too-many-positional-arguments
        ### Basic parameters
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.crpix1 = crpix1
        self.crpix2 = crpix2
        self.pos_angle_detector = pos_angle_detector
        self.pixel_scale = pixel_scale
        self.plate_scale = plate_scale
        self.xy_centers = xy_centers

        if isinstance(self.xy_centers, dict):
            ### Convert the dictionary of (x,y) centers to a table
            self.sca_list = np.array(
                sorted(list(self.xy_centers.keys())),
                dtype=int,
            )
            self.xy_centers_tbl = pd.DataFrame(
                self.xy_centers, index=["x", "y"]
            ).T
        elif isinstance(self.xy_centers, pd.DataFrame):
            self.sca_list = self.xy_centers.index.to_numpy()
            self.xy_centers_tbl = self.xy_centers
        else:
            raise Exception("Provide xy_centers as a dict or pd.Dataframe")

        ### Define the Polygons for each SCA
        self.sca_polygons = self.define_sca_polygons()

    def get_sca_center(self, sca):
        """
        Returns the center(s) for given SCAs
        """
        xcen = self.xy_centers_tbl.loc[sca, "x"]
        ycen = self.xy_centers_tbl.loc[sca, "y"]
        if len(np.atleast_1d(sca)) > 1:
            return xcen.to_numpy(), ycen.to_numpy()
        else:
            return xcen, ycen

    def define_sca_polygons(
        self, lw=2, facecolor="none", edgecolor="k", alpha=0.9, zorder=10
    ):  # pylint: disable=too-many-positional-arguments
        """
        Defines the SCA locations on the MPA (in mm) and their sizes (in pix)
        """
        sca_polygons = {}
        for sca in self.sca_list:

            xy_cen = self.xy_centers_tbl.loc[sca]
            xy_vrt = np.array(xy_cen) - (
                self.crpix1 / self.plate_scale,
                self.crpix2 / self.plate_scale,
            )
            dx = self.naxis1 / self.plate_scale
            dy = self.naxis2 / self.plate_scale

            sca_polygons[sca] = Rectangle(
                xy_vrt,
                dx,
                dy,
                angle=self.pos_angle_detector,
                rotation_point="center",
                lw=lw,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                zorder=zorder,
            )
        return sca_polygons

    def calculate_fpa_pos(
        self, ra, dec, pointing_ra, pointing_dec, pointing_pa
    ):  # pylint: disable=too-many-positional-arguments
        """
        Returns the FPA (x,y) position for given ra, dec and pointing
        The pointing_pa is defined as the orientation angle of the L2 image
        relative to the on-sky north-up, east-left WCS
        Units are in 'deg'
        """
        ra, dec = np.atleast_1d(ra), np.atleast_1d(dec)

        dx = (ra - pointing_ra) * np.cos(np.deg2rad(dec))
        dy = dec - pointing_dec
        xy = np.vstack([dx, dy])

        rot_matrix = self.get_pa_rotation(pa=pointing_pa)
        xy = rot_matrix @ xy

        xfpa = -xy[0, :]
        yfpa = -xy[1, :]

        return xfpa, yfpa
    
    def get_pa_rotation(self, pa):
        """
        Returns the rotation matrix for given rotation pa
        Units are in 'deg'
        """
        theta = np.deg2rad(
            pa + 180 - 60
        )  # on-sky PA to focal plane coordinate system
        return np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
    
    def match_pos_to_sca(self, xmpa, ympa):
        """
        Returns matching SCA in which the given point(s) lie
        Units are in 'mm'
        """
        xmpa, ympa = np.atleast_1d(xmpa), np.atleast_1d(ympa)
        pos = np.array([xmpa, ympa]).T
        cond = np.array(
            [
                self.sca_polygons[sca].contains_points(
                    self.sca_polygons[sca].get_data_transform().transform(pos)
                )
                for sca in self.sca_list
            ],
            dtype=bool,
        )
        j, i = np.where(cond)

        sca_match = np.zeros(len(xmpa), dtype=int)
        sca_match[i] = self.sca_list[j]

        if len(sca_match) > 1:
            raise Exception("Position matched to multiple SCAs!")
        elif sca_match == 0:
            warnings.warn("No matching SCA; returning closest one.")
            xdist = np.abs(xmpa - self.xy_centers_tbl["x"].to_numpy())
            sca_column = self.sca_list[np.argsort(xdist)[:3]]
            ydist = np.abs(
                ympa - self.xy_centers_tbl.loc[sca_column, "y"].to_numpy()
            )
            return sca_column[np.argmin(ydist)]
        else:
            return sca_match[0]

    def convert_sca_to_mpa(self, xsca, ysca, sca):
        """
        Returns MPA position [mm] for reference position in the SCA [pixel]
        Input: xsca, ysca are source position, in px, in detector plane (SCA)
        Output: xmpa, ympa are source position, in mm, in focal plane (MPA)
        """
        dx = (xsca - self.crpix1) * -1
        dy = ysca - self.crpix2

        # Rotation terms might be needed here
        xoff = dx / self.plate_scale
        yoff = dy / self.plate_scale

        xmpa = xoff + self.xy_centers_tbl.loc[sca, "x"]
        ympa = yoff + self.xy_centers_tbl.loc[sca, "y"]

        return xmpa, ympa

    def convert_mpa_to_sca(self, xmpa, ympa, sca=None):
        """
        Returns SCA position [pixel] for reference position in the MPA [mm]
        Input: xmpa, ympa are source position, in px, in detector plane (SCA)
        Output: xfpa, yfpa are source position, in deg, in focal plane (FPA)
        """
        if sca is None:
            sca = self.match_pos_to_sca(xmpa=xmpa, ympa=ympa)

        xcen, ycen = self.get_sca_center(sca=sca)
        xoff = xmpa - xcen
        yoff = ympa - ycen

        # Rotation terms might be needed here
        xrot = xoff * self.plate_scale
        yrot = yoff * self.plate_scale

        xdet = -xrot + self.crpix1
        ydet = yrot + self.crpix2

        return xdet, ydet

    def convert_sca_to_fpa(self, xsca, ysca, sca):
        """
        Returns FPA position [degree] for reference position in the SCA [px]
        Input: xsca, ysca are source position, in px, in detector plane (SCA)
        Output: xfpa, yfpa are source position, in deg, in focal plane (FPA)
        """
        dx = (xsca - self.crpix1) * -1
        dy = ysca - self.crpix2

        xcen, ycen = self.get_sca_center(sca=sca)
        xfpa = (xcen * self.plate_scale + dx) * self.pixel_scale / 3600
        yfpa = (ycen * self.plate_scale + dy) * self.pixel_scale / 3600

        return xfpa, yfpa

    def convert_fpa_to_sca(self, xfpa, yfpa, sca=None):
        """
        Returns SCA position [pixel] for reference position in the FPA [degree]
        Input: xfpa, yfpa are source position, in deg, in focal plane (FPA)
        Output: xsca, ysca are source position, in px, in detector plane (SCA)
        """
        if sca is None:
            xmpa, ympa = self.convert_fpa_to_mpa(xfpa=xfpa, yfpa=yfpa)
            sca = self.match_pos_to_sca(xmpa=xmpa, ympa=ympa)

        xcen, ycen = self.get_sca_center(sca=sca)
        dx = (xfpa * 3600 / self.pixel_scale) - (xcen * self.plate_scale)
        dy = (yfpa * 3600 / self.pixel_scale) - (ycen * self.plate_scale)

        xsca = -dx + self.crpix1
        ysca = dy + self.crpix2

        return xsca, ysca

    def convert_fpa_to_mpa(self, xfpa, yfpa):

        xmpa = xfpa * 3600 / self.pixel_scale / self.plate_scale
        ympa = yfpa * 3600 / self.pixel_scale / self.plate_scale

        return xmpa, ympa

    def convert_mpa_to_fpa(self, xmpa, ympa):

        xfpa = xmpa * self.plate_scale * self.pixel_scale / 3600
        yfpa = ympa * self.plate_scale * self.pixel_scale / 3600

        return xfpa, yfpa

    def generate_xyfpa_grid(self, npts, custom_slice=None):
        """
        Utility function to generate a grid of (x,y) in FPA coords
        """
        xref_fpa, yref_fpa, ref_sca = [], [], []
        for sca in self.sca_list:

            xref, yref = np.mgrid[
                0 : self.naxis1 : npts * 1j,
                0 : self.naxis2 : npts * 1j,
            ]

            if custom_slice is not None:
                xref = xref[custom_slice, custom_slice]
                yref = yref[custom_slice, custom_slice]

            xyref_fpa = self.convert_sca_to_fpa(
                xsca=xref.ravel(),
                ysca=yref.ravel(),
                sca=sca,
            )
            xref_fpa = np.concatenate([xref_fpa, xyref_fpa[0]])
            yref_fpa = np.concatenate([yref_fpa, xyref_fpa[1]])
            ref_sca = np.concatenate([ref_sca, [sca] * len(xyref_fpa[0])])

        return xref_fpa, yref_fpa, ref_sca


def transform_wl(wl, wl_reference, kind, inverse=False):
    """
    Helper function to transform the wllengths for use in the optical model
    """
    if kind == "linear":
        if not inverse:
            return wl - wl_reference
        else:
            return wl + wl_reference
    elif kind == "log":
        if not inverse:
            return np.log10(wl / wl_reference)
        else:
            return 10**wl * wl_reference
    else:
        raise Exception("Invalid kind provided: choose from 'linear' or 'log'.")
