"""Render strategy per (backend, source kind, order): what ``auto`` picks.

The driver renders stars and galaxies in separate batches, per order, so
each (kind, order) can use its own renderer and accumulation:

* ``segment_nodes``: None = the production deposit; an int = the
  trace-aligned segment render with that many wavelength nodes
  (``roman_disperser.segment_render``; 56 = all PSF grid wavelengths,
  which reproduces the deposit to float32 rounding);
* ``accumulate``: None = the sequential loop; ``"window"`` / ``"tiled"`` =
  per-source windows with a fixed sum order (``roman_disperser.accumulate``,
  deterministic on GPU but slower there, so ``auto`` never picks them; ask
  for them explicitly when run-to-run identical frames matter).

``auto`` asks for **speed**. The table holds the fastest measured choice;
host-side only, never inside jitted code. Measurements (ms per source,
production settings: interp, rebinning 40/4/2 A, star PSF hoist):

GPU, NVIDIA L40S (cluster T86 + newrd bench_seg, jax 0.11.2)::

    kind    order  loop   window  tiled  seg12+loop  seg12+win
    galaxy  0      0.583  1.134   1.896  0.367       0.308
    galaxy  1      0.932  1.587   2.665  0.509       1.491
    galaxy  2      1.246  2.060   3.571  1.108       4.059
    star    0      0.209  0.201   0.422  0.226       0.139
    star    1      0.331  0.302   0.616  0.326       0.693
    star    2      0.459  0.458   0.808  0.508       1.794

Node count (L40S, newrd bench_seg: 1000 stars / 200 galaxies per order,
production batches; accuracy = max |image - loop image| / peak)::

    kind    order  loop   seg12          seg28          seg56
    galaxy  0      0.519  0.278 (2.3e-4) 0.366 (3.0e-5) 0.637 (6e-7)
    galaxy  1      0.890  0.442 (3.1e-4) 0.707 (9.6e-5) 0.962 (2.8e-6)
    galaxy  2      1.207  1.101 (1.4e-4) 1.533 (2.6e-4) 1.265 (2.6e-6)
    star    0      0.173  0.187          0.216          0.281
    star    1      0.303  0.284          0.310          0.433
    star    2      0.420  0.484          0.474          0.613

The same ranking holds on an A40 and an RTX A5000 (T86 logs; galaxies
seg12+loop 2.0-2.7x faster in orders 0-1, 1.1-1.2x in order 2; stars:
the loop within a few % of the best everywhere), so one GPU row serves all
classes measured. On GPU, then, galaxies take the 12-node segment render
with the loop (2x faster in orders 0-1). **This is an approximation**: the
image differs from the deposit by up to ~3e-4 of the peak (flux within
6e-6), against ~2e-4 between the CPU and GPU deposit; 56 nodes would be
exact but are no faster than the loop. Stars keep the production deposit
with the loop. CPU (16 cores, BDI SCA 5 end to end):
default 51 s, 56-node segment render 30 s, 12 nodes 22 s, tiled 113 s;
the CPU keeps the 56-node segment render with the loop for both kinds (the
choice ``auto`` made before the per-kind table).
"""

from dataclasses import dataclass

KINDS = ("star", "galaxy")

# All PSF grid wavelengths: the segment render then reproduces the deposit.
EXACT_SEGMENT_NODES = 56


@dataclass(frozen=True)
class Strategy:
    """How one (kind, order) is rendered. See the module docstring."""
    segment_nodes: object = None   # None or int >= 2
    accumulate: object = None      # None, "window" or "tiled"


# Galaxy segment node count per order on GPU: the fastest measured (see the
# module docstring for speed and accuracy). Orders not listed take
# EXACT_SEGMENT_NODES.
GPU_GALAXY_NODES = {"0": 12, "1": 12, "2": 12}

_CPU = Strategy(segment_nodes=EXACT_SEGMENT_NODES)
_GPU_STAR = Strategy()


def strategy(backend, kind, order):
    """The ``auto`` :class:`Strategy` for one (backend, kind, order).

    ``backend`` is ``jax.default_backend()`` ("cpu", "gpu", "cuda", "rocm",
    ...): "cpu" uses the CPU row, anything else the GPU rows.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if backend == "cpu":
        return _CPU
    if kind == "star":
        return _GPU_STAR
    return Strategy(segment_nodes=GPU_GALAXY_NODES.get(
        str(order), EXACT_SEGMENT_NODES))


def auto_table(backend, orders):
    """``{kind: {order: Strategy}}`` for ``auto`` on this backend."""
    return {kind: {str(o): strategy(backend, kind, o) for o in orders}
            for kind in KINDS}
