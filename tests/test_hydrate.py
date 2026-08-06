"""Tests for the vendored-data hydrator (roman_disperser.hydrate).

Network-free: the GitHub discovery/download primitives are exercised either
via a local ``file://`` URL or by monkeypatching, so the suite stays offline
and fast.
"""

import json
import tarfile

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


def test_extract_asset_up_to_date_skips(tmp_path, monkeypatch):
    # Marker present AND lock agrees with the resolved tag -> nothing fetched.
    monkeypatch.setattr(
        hydrate, "list_release_files",
        lambda tag: [("sensitivities.tar.gz", "file:///nonexistent")],
    )
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


def test_main_rejects_unknown_asset(tmp_path):
    try:
        main(["--only", "bogus", "--dest", str(tmp_path)])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for unknown asset")
