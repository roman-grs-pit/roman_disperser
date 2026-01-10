"""Roman Optical Model"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.interpolate import LinearNDInterpolator

from .optical_model_utils import RomanDetectorCoordinates, transform_wl


class RomanOpticalModel:
    """
    Unit and label system used --
    SCA: Sensor Chip Assembly (Invid. Detectors); position defined in [pixels]
    FPA: Focal Plane Assembly; position defined in [deg] from center of the SCA
    MPA: Mosaic Plate Assembly; position defined in [mm]
    """

    def __init__(self, config_file=None, old_format=False):

        self.old_format = old_format

        # Load the specified model file (YAML format)
        self.config_file = config_file
        self.load_model(config_file=config_file)

        # Vectorize functions
        self.get_beam_coeff_vec = np.vectorize(self._get_beam_coeff)
        self.get_beam_trace_vec = np.vectorize(self._get_beam_trace)

        self.get_beam_coeff = lambda *args, **kwargs: np.atleast_1d(
            self.get_beam_coeff_vec(*args, **kwargs)
        )
        self.get_beam_trace = lambda *args, **kwargs: np.atleast_1d(
            self.get_beam_trace_vec(*args, **kwargs)
        )

    def load_model(self, config_file=None):
        """
        Reads in the configuration file for the optical model
        """
        # Read in the YAML config file
        if config_file is not None:
            self.config_file = config_file

        with open(self.config_file, mode="r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.meta = self.config["roman"]["meta"]
        self.detmod = self.config["roman"]["detector_model"]
        self.optmod = self.config["roman"]["optical_model"]

        self.coords = RomanDetectorCoordinates(**self.detmod)
        self.sca_list = self.coords.sca_list

        # Define some default parameters
        self.optical_element = self.meta["optical_element"]

        # Define wavelength parameters
        self.wl_transform = self.optmod["wl_transform"]
        self.wl_reference = self.optmod["wl_reference"]
        self.wl_min = self.optmod["wl_min"]
        self.wl_max = self.optmod["wl_max"]

        # Define the wavelength resolution of the returned beam trace info
        self.wl_grid = np.linspace(self.wl_min, self.wl_max, 20)

        # Define the polynomial orders
        self.dimen_map = self.optmod["dimen_map"]
        self.dimen_crv = self.optmod["dimen_crv"]
        self.dimen_ids = self.optmod["dimen_ids"]

        # List of defined orders
        self.orders_defined = self.optmod["orders_defined"]

        # Cast the polynomial coefficients as arrays
        self.beam_coeffs = {}
        for order in self.orders_defined:
            if order in self.optmod["orders"]:
                self.beam_coeffs[order] = {}
                self.beam_coeffs[order]["X_ij"] = np.array(
                    self.optmod["orders"][order]["xmap_ij_coeff"],
                    dtype=float,
                )
                self.beam_coeffs[order]["Y_ij"] = np.array(
                    self.optmod["orders"][order]["ymap_ij_coeff"],
                    dtype=float,
                )
                self.beam_coeffs[order]["C_ijk"] = np.array(
                    self.optmod["orders"][order]["crv_ijk_coeff"],
                    dtype=float,
                )
                self.beam_coeffs[order]["D_ijk"] = np.array(
                    self.optmod["orders"][order]["ids_ijk_coeff"],
                    dtype=float,
                )

    def get_map_coords(self, xfpa, yfpa, order="1"):
        """
        Method to return the trace offset location for a given reference pixel
        Trace offset location returned defines the "wl_reference" wavelength
        Input: reference position in deg from get_FPA_coords
        Output: trace offset position in mm
        """
        x = np.atleast_1d(xfpa)[:, np.newaxis] ** np.arange(self.dimen_map["i"])
        y = np.atleast_1d(yfpa)[:, np.newaxis] ** np.arange(self.dimen_map["j"])

        xmpa = np.diagonal(x @ self.beam_coeffs[order]["X_ij"] @ y.T)
        ympa = np.diagonal(x @ self.beam_coeffs[order]["Y_ij"] @ y.T)

        return xmpa, ympa

    def get_trace_coeffs(self, xfpa, yfpa, order="1"):
        """
        Method to return the curvature and inverse dispersion soln coefficients
        for a given trace offset location
        Input: trace offset location from get_map_coords() and trace length
        Output: curvature and inverse dispersion soln coefficients
        """
        x = np.atleast_1d(xfpa)[:, np.newaxis] ** np.arange(self.dimen_crv["j"])
        y = np.atleast_1d(yfpa)[:, np.newaxis] ** np.arange(self.dimen_crv["k"])
        crv = np.diagonal(
            x @ self.beam_coeffs[order]["C_ijk"] @ y.T, axis1=1, axis2=2
        )

        x = np.atleast_1d(xfpa)[:, np.newaxis] ** np.arange(self.dimen_ids["j"])
        y = np.atleast_1d(yfpa)[:, np.newaxis] ** np.arange(self.dimen_ids["k"])
        ids = np.diagonal(
            x @ self.beam_coeffs[order]["D_ijk"] @ y.T, axis1=1, axis2=2
        )

        return crv, ids

    def integerize_width(self, width):
        """
        Utility to convert the width to a valid integer (rounds up)
        """
        return int(np.ceil(width))

    def _get_beam_coeff(
        self, xref_fpa, yref_fpa, sca=None, width=1, order="1"
    ):  # pylint: disable=too-many-positional-arguments
        """
        Computes the reference pixel location of the trace for the source
        position as well as the wavelength soln and curvature coefficients
        """
        # Compute the source positions in SCA/MPA coordinates
        xref_mpa, yref_mpa = self.coords.convert_fpa_to_mpa(
            xfpa=xref_fpa, yfpa=yref_fpa
        )
        if sca is None:
            sca = self.coords.match_pos_to_sca(xmpa=xref_mpa, ympa=yref_mpa)
        xref_sca, yref_sca = self.coords.convert_fpa_to_sca(
            xref_fpa, yref_fpa, sca=sca
        )

        # Setup the reference position according to the specified width
        width = self.integerize_width(width=width)
        xref_sca_arr = xref_sca - np.floor(width / 2) + np.arange(width)
        yref_sca_arr = np.full(width, yref_sca)

        # Initial transform to FPA coordinates for the transformation soln
        xfpa, yfpa = self.coords.convert_sca_to_fpa(
            xsca=xref_sca_arr, ysca=yref_sca_arr, sca=sca
        )

        # Apply transformation soln to get the reference position of the trace
        xmpa, ympa = self.get_map_coords(xfpa=xfpa, yfpa=yfpa, order=order)

        # Transform the reference position of the trace back to pixels
        xsca, ysca = self.coords.convert_mpa_to_sca(
            xmpa=xmpa, ympa=ympa, sca=sca
        )

        # Apply the wavelength and curvature soln to get the trace
        if not self.old_format:
            crv, ids = self.get_trace_coeffs(xfpa=xfpa, yfpa=yfpa, order=order)
        else:
            crv, ids = self.get_trace_coeffs(xfpa=xmpa, yfpa=xmpa, order=order)

        return {
            # Input values
            "optical_element": self.optical_element,
            "order": order,
            "ref_pos_fpa": [xref_fpa, yref_fpa],
            "ref_pos_mpa": [xref_mpa, yref_mpa],
            "ref_pos_sca": [xref_sca, yref_sca],
            "sca": sca,
            "width": width,
            # Reference positions of the trace
            "ref_trace_fpa": np.array(
                [self.coords.convert_mpa_to_fpa(xmpa=xmpa, ympa=ympa)]
            ).T,
            "ref_trace_mpa": np.array([xmpa, ympa]).T,
            "ref_trace_sca": np.array([xsca, ysca]).T,
            # Coefficients for the curvature and wavelength soln
            "crv": crv,
            "ids": ids,
        }

    def _get_beam_trace(
        self, xref_fpa, yref_fpa, sca=None, width=1, order="1"
    ):  # pylint: disable=too-many-positional-arguments
        """
        Main workhorse function to produce the full trace (x, y, wavelength)
        using the coefficients from _get_beam_coeff()
        """
        # Ensure width is an integer
        width = self.integerize_width(width=width)

        # Get the required coefficients
        coeff = self._get_beam_coeff(
            xref_fpa=xref_fpa,
            yref_fpa=yref_fpa,
            sca=sca,
            width=width,
            order=order,
        )

        # Setup the wavelength array
        wl = transform_wl(
            wl=self.wl_grid,
            wl_reference=self.wl_reference,
            kind=self.wl_transform.lower(),
        )

        # Get the position along the trace
        w = wl[:, np.newaxis] ** np.arange(self.dimen_ids["i"])
        dely = w @ coeff["ids"]

        # Get the curvature along the trace
        y = dely[:, :, np.newaxis] ** np.arange(self.dimen_crv["i"])
        delx = np.diagonal(y @ coeff["crv"], axis1=1, axis2=2)

        if self.old_format:
            delx = delx / self.detmod["plate_scale"]
            dely = dely / self.detmod["plate_scale"]

        # Populate the relevant trace info
        coeff["trace_mpa_x"] = (
            coeff["ref_trace_mpa"][:, 0][:, np.newaxis] + delx.T
        )
        coeff["trace_mpa_y"] = (
            coeff["ref_trace_mpa"][:, 1][:, np.newaxis] + dely.T
        )
        coeff["trace_fpa_x"], coeff["trace_fpa_y"] = (
            self.coords.convert_mpa_to_fpa(
                xmpa=coeff["trace_mpa_x"],
                ympa=coeff["trace_mpa_y"],
            )
        )
        coeff["trace_sca_x"], coeff["trace_sca_y"] = (
            self.coords.convert_mpa_to_sca(
                xmpa=coeff["trace_mpa_x"],
                ympa=coeff["trace_mpa_y"],
                sca=coeff["sca"],
            )
        )

        coeff["trace_wvl"] = np.full((width, len(wl)), self.wl_grid)

        return coeff

    def get_beam_trace_wl(
        self, xpos, ypos, xref_fpa, yref_fpa, sca=None, width=25, order="1"
    ):  # pylint: disable=too-many-positional-arguments
        """
        Inverts beam trace solution returning wavelength at a given pixel
        for a reference position and source width
        """
        coeff = self._get_beam_trace(
            xref_fpa=xref_fpa,
            yref_fpa=yref_fpa,
            sca=sca,
            width=width,
            order=order,
        )

        interp = LinearNDInterpolator(
            (coeff["trace_pix_x"].ravel(), coeff["trace_pix_y"].ravel()),
            coeff["trace_wvl"].ravel(),
            fill_value=np.nan,
        )

        return interp(xpos, ypos)

    def plot_quick_look(self, width=1, order="1", cmap=plt.cm.Spectral_r):
        """
        Quick-view plot to visualize the optical model in the MPA coordinates
        """

        fig, ax = plt.subplots(
            1, 1, figsize=(11.5, 8), dpi=75, tight_layout=True
        )

        ax.text(
            0.5,
            0.98,
            f"{self.meta["optical_element"].upper():s}" f" - order {order:s}",
            fontsize=20,
            fontweight=600,
            color="k",
            va="top",
            ha="center",
            transform=ax.transAxes,
        )

        norm = Normalize(vmin=self.wl_min, vmax=self.wl_max)
        ax.set_xlim(140, -140)
        ax.set_ylim(-120, 75)
        ax.set_aspect(1)

        for sca in self.sca_list:
            ax.add_patch(copy(self.coords.sca_polygons[sca]))
            ax.text(
                *self.detmod["xy_centers"][sca],
                f"SCA{sca:d}",
                fontsize=18,
                fontweight=600,
                va="center",
                ha="center",
                color="dimgray",
                alpha=0.3,
            )

        ### Generate a grid of test points
        xref_fpa, yref_fpa, ref_sca = self.coords.generate_xyfpa_grid(
            npts=9, custom_slice=slice(1, 9, 3)
        )

        for sca in self.sca_list:
            cond = ref_sca == sca
            coeffs = self.get_beam_trace(
                xref_fpa=xref_fpa[cond],
                yref_fpa=yref_fpa[cond],
                sca=sca,
                width=width,
                order=order,
            )
            for coeff in coeffs:
                ### Show the source position
                ax.scatter(
                    *coeff["ref_pos_mpa"],
                    marker="*",
                    facecolor="none",
                    edgecolor="k",
                    s=200,
                    zorder=10,
                )

                ### Show the trace position @ reference wavelength
                ax.scatter(
                    *coeff["ref_trace_mpa"].T,
                    marker="o",
                    color="k",
                    s=15,
                    zorder=10,
                )

                ### Show the trace
                trace_xy = np.array(
                    [coeff["trace_mpa_x"], coeff["trace_mpa_y"]]
                ).T.reshape((-1, 1, 2))
                trace_seg = np.concatenate(
                    [trace_xy[:-1], trace_xy[1:]], axis=1
                )
                lc = LineCollection(
                    trace_seg,
                    cmap=cmap,
                    norm=norm,
                    linewidth=3,
                    zorder=8,
                )
                lc.set_array(coeff["trace_wvl"].ravel())
                ax.add_collection(lc)

        return fig, ax
