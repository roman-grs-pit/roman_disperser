"""Tests for the vendored-data hydrator (roman_disperser.hydrate).

Network-free: the GitHub discovery/download primitives are exercised either
via a local ``file://`` URL or by monkeypatching, so the suite stays offline
and fast.
"""

import json

from roman_disperser import hydrate
from roman_disperser.hydrate import (
    _download,
    _filter_sca,
    main,
    resolve_manifest,
    write_lock,
)


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


def test_main_rejects_unknown_asset(tmp_path):
    try:
        main(["--only", "bogus", "--dest", str(tmp_path)])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for unknown asset")
