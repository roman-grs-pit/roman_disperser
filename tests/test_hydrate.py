"""Tests for the vendored-data hydrator (roman_disperser.hydrate).

Network-free: the GitHub discovery/download primitives are exercised either
via a local ``file://`` URL or by monkeypatching, so the suite stays offline
and fast.
"""

import json
import tarfile

import pytest

from roman_disperser import hydrate
from roman_disperser.hydrate import (
    _download,
    _filter_sca,
    main,
    resolve_manifest,
    write_lock,
)


def _make_tarball(tmp_path, name, files):
    """Build a small .tar.gz of {filename: content} and return its file:// URL."""
    stage = tmp_path / f"stage-{name}"
    tar_path = tmp_path / name
    with tarfile.open(tar_path, "w:gz") as tar:
        for fname, content in files.items():
            p = stage / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            tar.add(p, arcname=fname)
    return tar_path.as_uri()


def _psf_files():
    return (
        [(f"psf_WFI{n:02d}_GRISM0_x.npz", "u") for n in range(1, 19)]
        + [(f"psf_WFI{n:02d}_GRISM1_x.npz", "u") for n in range(1, 19)]
    )


def test_filter_sca_selects_requested():
    out = _filter_sca(_psf_files(), [1, 5])
    assert {n for n, _ in out} == {
        "psf_WFI01_GRISM0_x.npz", "psf_WFI01_GRISM1_x.npz",
        "psf_WFI05_GRISM0_x.npz", "psf_WFI05_GRISM1_x.npz",
    }


def test_filter_sca_none_is_passthrough():
    files = _psf_files()
    assert _filter_sca(files, None) == files


def test_resolve_manifest_from_lock(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"psf": "psf-v9", "catalog": "catalog-v9"}))
    versions, source = resolve_manifest(lock=str(lock))
    assert versions["psf"] == "psf-v9"
    assert source.startswith("lock:")


def test_resolve_manifest_falls_back_when_remote_unavailable(monkeypatch):
    monkeypatch.setattr(hydrate, "_get_json",
                        lambda url: (_ for _ in ()).throw(RuntimeError("offline")))
    versions, source = resolve_manifest()
    assert versions == hydrate.DEFAULT_MANIFEST
    assert source == "default"


def test_write_lock_merges_existing(tmp_path):
    (tmp_path / hydrate.LOCK_NAME).write_text(json.dumps({"psf": "psf-v1"}))
    write_lock(tmp_path, {"catalog": "catalog-v2"})
    data = json.loads((tmp_path / hydrate.LOCK_NAME).read_text())
    assert data == {"psf": "psf-v1", "catalog": "catalog-v2"}


def test_download_atomic_skip_and_force(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    dest = tmp_path / "out" / "f.bin"

    assert _download(src.as_uri(), dest) is True
    assert dest.read_bytes() == b"hello"
    # idempotent: skip when present
    assert _download(src.as_uri(), dest) is False
    # force re-download
    assert _download(src.as_uri(), dest, force=True) is True
    # no temp file left behind
    assert not (dest.parent / (dest.name + ".part")).exists()


def test_main_with_lock_writes_resolved_lock(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(hydrate, "hydrate_asset",
                        lambda *a, **k: captured.append((a, k)))
    in_lock = tmp_path / "in.lock"
    in_lock.write_text(json.dumps({"psf": "psf-vX", "catalog": "catalog-vY"}))

    rc = main(["--lock", str(in_lock), "--dest", str(tmp_path / "data")])

    assert rc == 0
    assert len(captured) == 2  # both assets hydrated
    out_lock = json.loads((tmp_path / "data" / hydrate.LOCK_NAME).read_text())
    assert out_lock == {"psf": "psf-vX", "catalog": "catalog-vY"}


def test_main_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(hydrate, "hydrate_asset", lambda *a, **k: None)
    in_lock = tmp_path / "in.lock"
    in_lock.write_text(json.dumps({"psf": "psf-vX"}))

    rc = main(["--lock", str(in_lock), "--dest", str(tmp_path / "data"),
               "--only", "psf", "--dry-run"])

    assert rc == 0
    assert not (tmp_path / "data" / hydrate.LOCK_NAME).exists()


def test_hydrate_asset_downloads_to_destination(tmp_path, monkeypatch):
    # Exercises the real hydrate_asset -> _download call site (a swapped
    # url/dest there is invisible to the direct _download test).
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("Roman_grism_OpticalModel_v0.8.yaml", src.as_uri())],
    )
    base = tmp_path / "data"
    hydrate.hydrate_asset(hydrate.ASSETS["optical_model"], "optical-model-v0.8", base)
    assert (base / "Roman_grism_OpticalModel_v0.8.yaml").read_bytes() == b"payload"


def test_nonextract_asset_refetches_on_lock_mismatch(tmp_path, monkeypatch):
    # Non-extract filenames need not carry the release version (PSF caches
    # don't), so a present file may be another delivery's under the same
    # name. A lock entry disagreeing with the resolved tag must force a
    # re-download — otherwise --update writes the new tag into the lock
    # while the old bytes stay on disk.
    src = tmp_path / "src.bin"
    src.write_bytes(b"new delivery")
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("psf_WFI01_GRISM0_4x4x56.npz", src.as_uri())],
    )
    base = tmp_path / "data"
    dest = base / "psf_cache" / "psf_WFI01_GRISM0_4x4x56.npz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old delivery")
    hydrate.hydrate_asset(hydrate.ASSETS["psf"], "psf-v2", base,
                          locked_tag="psf-v1")
    assert dest.read_bytes() == b"new delivery"


def test_nonextract_asset_pinned_present_not_refetched(tmp_path, monkeypatch):
    # Lock agrees with the resolved tag -> present files are left alone
    # (the URL below would fail if a fetch were attempted).
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("psf_WFI01_GRISM0_4x4x56.npz", "file:///nonexistent")],
    )
    base = tmp_path / "data"
    dest = base / "psf_cache" / "psf_WFI01_GRISM0_4x4x56.npz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"installed")
    hydrate.hydrate_asset(hydrate.ASSETS["psf"], "psf-v1", base,
                          locked_tag="psf-v1")
    assert dest.read_bytes() == b"installed"


def test_nonextract_sca_filtered_upgrade_refused(tmp_path, monkeypatch):
    # An --sca-filtered install at a tag the lock disagrees with must be
    # refused when files outside the filter are present: only the filtered
    # files would be re-fetched, but the lock would record the new tag for
    # all of them — a lying lock about 35 of 36 files.
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("psf_WFI01_GRISM0_x.npz", "file:///unused"),
                     ("psf_WFI02_GRISM0_x.npz", "file:///unused")],
    )
    base = tmp_path / "data"
    (base / "psf_cache").mkdir(parents=True)
    (base / "psf_cache" / "psf_WFI02_GRISM0_x.npz").write_bytes(b"v1 bytes")
    with pytest.raises(ValueError, match="Refusing an --sca-filtered"):
        hydrate.hydrate_asset(hydrate.ASSETS["psf"], "psf-v2", base,
                              sca=[1], locked_tag="psf-v1")

    # Fresh dir (nothing outside the filter): the same call is fine.
    base2 = tmp_path / "data2"
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("psf_WFI01_GRISM0_x.npz", src.as_uri()),
                     ("psf_WFI02_GRISM0_x.npz", src.as_uri())],
    )
    hydrate.hydrate_asset(hydrate.ASSETS["psf"], "psf-v2", base2,
                          sca=[1], locked_tag=None)
    assert (base2 / "psf_cache" / "psf_WFI01_GRISM0_x.npz").exists()
    assert not (base2 / "psf_cache" / "psf_WFI02_GRISM0_x.npz").exists()


def test_default_manifest_covers_all_assets():
    # The offline fallback must know every registered asset, or a fresh
    # hydrate on a box that cannot reach GitHub silently installs a subset.
    assert set(hydrate.DEFAULT_MANIFEST) == set(hydrate.ASSETS)


def _raiser(msg):
    def fail(*a, **k):
        raise AssertionError(msg)
    return fail


def test_extract_asset_up_to_date_skips(tmp_path, monkeypatch):
    # Marker present AND lock agrees with the resolved tag -> nothing fetched,
    # not even the release listing (the check precedes any network call).
    monkeypatch.setattr(hydrate, "list_release_files", _raiser("network hit"))
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("v1 contents")

    hydrate.hydrate_asset(asset, "sensitivities-v1", base,
                          locked_tag="sensitivities-v1")
    assert (base / asset.subdir / asset.done_marker).read_text() == "v1 contents"


def test_extract_asset_reinstalls_on_version_change(tmp_path, monkeypatch):
    # Marker present but the lock records a DIFFERENT version: the done-marker
    # alone must not veto the install (it only proves *some* version was
    # extracted once), so the new tarball is fetched and extracted.
    url = _make_tarball(tmp_path, "sens.tar.gz",
                        {"sensitivity_map.yaml": "v2 contents"})
    monkeypatch.setattr(hydrate, "list_release_files",
                        lambda tag: [("sens.tar.gz", url)])
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("v1 contents")

    hydrate.hydrate_asset(asset, "sensitivities-v2", base,
                          locked_tag="sensitivities-v1")
    assert (base / asset.subdir / asset.done_marker).read_text() == "v2 contents"


def test_extract_asset_reinstalls_when_lock_silent(tmp_path, monkeypatch):
    # Marker present but no lock entry (pre-lock or hand-assembled data dir):
    # the installed version is unknown, so reinstall rather than let the lock
    # written afterwards claim a version that was never verified on disk.
    url = _make_tarball(tmp_path, "sens.tar.gz",
                        {"sensitivity_map.yaml": "v1 contents"})
    monkeypatch.setattr(hydrate, "list_release_files",
                        lambda tag: [("sens.tar.gz", url)])
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("unknown contents")

    hydrate.hydrate_asset(asset, "sensitivities-v1", base, locked_tag=None)
    assert (base / asset.subdir / asset.done_marker).read_text() == "v1 contents"


def test_main_lock_matches_contents_after_manifest_bump(tmp_path, monkeypatch):
    # End-to-end regression for the lock/contents divergence: a data dir at
    # v1 re-hydrated against a manifest that moved to v2 must end with BOTH
    # the v2 contents and a lock saying v2 (previously it skipped the install
    # on the done-marker yet still rewrote the lock to v2).
    url = _make_tarball(tmp_path, "sens.tar.gz",
                        {"sensitivity_map.yaml": "v2 contents"})
    monkeypatch.setattr(hydrate, "list_release_files",
                        lambda tag: [("sens.tar.gz", url)])
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("v1 contents")
    write_lock(base, {"sensitivities": "sensitivities-v1"})
    manifest = tmp_path / "in.lock"
    manifest.write_text(json.dumps({"sensitivities": "sensitivities-v2"}))

    rc = main(["--lock", str(manifest), "--dest", str(base),
               "--only", "sensitivities"])

    assert rc == 0
    assert (base / asset.subdir / asset.done_marker).read_text() == "v2 contents"
    lock = json.loads((base / hydrate.LOCK_NAME).read_text())
    assert lock["sensitivities"] == "sensitivities-v2"


def test_main_plain_rehydrate_is_pinned_and_offline(tmp_path, monkeypatch):
    # A hydrated dir re-run with no version flags stays at its locked
    # versions: the manifest is never consulted, nothing is fetched, and the
    # lock is unchanged — even if the remote manifest has moved on. This is
    # the "a mistaken re-hydrate must not move science data" guarantee.
    monkeypatch.setattr(hydrate, "resolve_manifest", _raiser("manifest consulted"))
    monkeypatch.setattr(hydrate, "list_release_files", _raiser("network hit"))
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("v1 contents")
    write_lock(base, {"sensitivities": "sensitivities-v1"})

    rc = main(["--dest", str(base), "--only", "sensitivities"])

    assert rc == 0
    assert (base / asset.subdir / asset.done_marker).read_text() == "v1 contents"
    lock = json.loads((base / hydrate.LOCK_NAME).read_text())
    assert lock == {"sensitivities": "sensitivities-v1"}


def test_main_update_flag_moves_to_manifest(tmp_path, monkeypatch):
    # --update is the explicit opt-in: versions come from the manifest, the
    # changed asset is re-installed, and the lock is re-pinned.
    url = _make_tarball(tmp_path, "sens.tar.gz",
                        {"sensitivity_map.yaml": "v2 contents"})
    monkeypatch.setattr(
        hydrate, "resolve_manifest",
        lambda *a, **k: ({"sensitivities": "sensitivities-v2"}, "manifest:current"),
    )
    monkeypatch.setattr(hydrate, "list_release_files",
                        lambda tag: [("sens.tar.gz", url)])
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("v1 contents")
    write_lock(base, {"sensitivities": "sensitivities-v1"})

    rc = main(["--dest", str(base), "--only", "sensitivities", "--update"])

    assert rc == 0
    assert (base / asset.subdir / asset.done_marker).read_text() == "v2 contents"
    lock = json.loads((base / hydrate.LOCK_NAME).read_text())
    assert lock["sensitivities"] == "sensitivities-v2"


def test_main_pinned_completes_missing_assets(tmp_path, monkeypatch):
    # Pinned mode still installs assets the lock does not know yet (a newly
    # published asset, or a widened --only) at manifest versions, while
    # leaving pinned assets untouched at their locked versions.
    src = tmp_path / "om.yaml"
    src.write_text("model")
    monkeypatch.setattr(
        hydrate, "resolve_manifest",
        lambda *a, **k: ({"optical_model": "optical-model-v0.9",
                          "sensitivities": "sensitivities-v9"}, "manifest:current"),
    )
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("Roman_grism_OpticalModel_v0.9.yaml", src.as_uri())],
    )
    base = tmp_path / "data"
    asset = hydrate.ASSETS["sensitivities"]
    (base / asset.subdir).mkdir(parents=True)
    (base / asset.subdir / asset.done_marker).write_text("v1 contents")
    write_lock(base, {"sensitivities": "sensitivities-v1"})

    rc = main(["--dest", str(base), "--only", "optical_model,sensitivities"])

    assert rc == 0
    assert (base / "Roman_grism_OpticalModel_v0.9.yaml").read_text() == "model"
    assert (base / asset.subdir / asset.done_marker).read_text() == "v1 contents"
    lock = json.loads((base / hydrate.LOCK_NAME).read_text())
    assert lock == {"optical_model": "optical-model-v0.9",
                    "sensitivities": "sensitivities-v1"}


def test_main_rejects_unknown_asset(tmp_path):
    try:
        main(["--only", "bogus", "--dest", str(tmp_path)])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for unknown asset")
