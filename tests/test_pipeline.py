"""Tests for pipeline utilities: per-SCA RNG keys and FITS provenance.

These cover the v0.13.0 reproducibility changes (issue #20 and the
CODEVER/GITSHA provenance cards). Before v0.13.0 nothing exercised the RNG
path at all, which is how the list-position key bug shipped unnoticed.
"""

import re

import jax
import numpy as np
import pytest
from astropy.io import fits

from roman_disperser.pipeline import (
    get_code_version,
    get_git_sha,
    make_sca_keys,
    write_fits,
)


def _key_bits(key):
    return tuple(np.asarray(jax.random.key_data(key)).tolist())


class TestMakeScaKeys:
    def test_subset_invariance(self):
        """A subset run must reproduce the full run's key for shared SCAs.

        This is the issue #20 fix: keys depend on the SCA number, never on
        the position in (or the contents of) the SCA list.
        """
        pk = jax.random.key(42)
        full = make_sca_keys(pk, range(1, 19))
        for subset in ([5], [18], [3, 9, 15], [9, 3, 15]):
            keys = make_sca_keys(pk, subset)
            for sca in subset:
                assert _key_bits(keys[sca]) == _key_bits(full[sca])

    def test_keys_distinct_across_scas(self):
        pk = jax.random.key(42)
        keys = make_sca_keys(pk, range(1, 19))
        assert len({_key_bits(k) for k in keys.values()}) == 18

    def test_keys_distinct_across_pointings(self):
        a = make_sca_keys(jax.random.key(1), [5])
        b = make_sca_keys(jax.random.key(2), [5])
        assert _key_bits(a[5]) != _key_bits(b[5])

    def test_matches_fold_in(self):
        """Pin the derivation: fold_in(pointing_key, sca_num), nothing else."""
        pk = jax.random.key(7)
        keys = make_sca_keys(pk, [11])
        assert _key_bits(keys[11]) == _key_bits(jax.random.fold_in(pk, 11))


class TestProvenance:
    def test_code_version_matches_package(self):
        import importlib.metadata

        assert get_code_version() == importlib.metadata.version(
            "roman_disperser"
        )

    def test_dunder_version_matches_code_version(self):
        # __version__, get_code_version() and the CODEVER FITS card must be
        # one number; all three read the installed metadata.
        import roman_disperser

        assert roman_disperser.__version__ == get_code_version()

    def test_pipeline_module_is_exported(self):
        # The docs and notebooks tell users to reach for rd.pipeline; it
        # must be importable off the package like every other module.
        import roman_disperser

        assert hasattr(roman_disperser, "pipeline")
        assert "pipeline" in roman_disperser.__all__

    def test_git_sha_format(self):
        """40-hex SHA, optionally -dirty; or 'unknown' outside a checkout."""
        sha = get_git_sha()
        assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?|unknown", sha)


class TestWriteFitsHeaders:
    @pytest.fixture()
    def written(self, tmp_path):
        img = np.zeros((4, 4), dtype=np.float32)
        path = tmp_path / "out.fits"
        write_fits(
            img, img, str(path),
            pointing_ra=9.5, pointing_dec=0.95, pointing_pa=0.0,
            sca=5, exptime=190.22,
            rng_key_data=np.array([123, 456], dtype=np.uint32), seed=42,
            extra_headers={"MA_TABLE": (1036, "MA table number")},
        )
        with fits.open(path) as hdul:
            yield hdul[0].header

    def test_provenance_cards_present(self, written):
        """CODEVER/GITSHA are written unconditionally, in every mode."""
        assert written["CODEVER"] == get_code_version()
        assert written["GITSHA"] == get_git_sha()

    def test_rng_cards_roundtrip(self, written):
        assert written["SEED"] == 42
        assert written["RNDSEED0"] == 123
        assert written["RNDSEED1"] == 456

    def test_extra_headers_still_applied(self, written):
        assert written["MA_TABLE"] == 1036
