#!/usr/bin/env python
"""
Generate PSF caches for all detectors and orders of a dispersing element.

This script pre-generates all PSF payloads needed for full-field validation.
Grism defaults (4×4×56 grid, 18 SCAs × 2 STPSF filters = 36 payloads) take
approximately 3-4 hours with 1 worker, or ~30-40 minutes with 8 workers; the
prism (18 SCAs × 1 filter) takes half that.

Usage:
    pixi run python scripts/generate_psf_caches.py
    pixi run python scripts/generate_psf_caches.py --element prism --workers 8

Options:
    --cache-dir PATH    Cache directory (default: data/psf_cache)
    --element NAME      Dispersing element, grism or prism (default: grism)
    --orders ORDERS     Comma-separated orders (default: the element's orders
                        with a distinct STPSF filter: 0,1 grism; 1 prism)
    --detectors DETS    Comma-separated detectors (default: all 18)
    --workers N         Number of parallel workers (default: 1)
    --no-skip           Regenerate even if cache exists
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress noisy STPSF logging and warnings before importing
logging.getLogger('stpsf').setLevel(logging.ERROR)
logging.getLogger('poppy').setLevel(logging.ERROR)
logging.getLogger('webbpsf').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*outside the range.*')
warnings.filterwarnings('ignore', message='.*Attempted to get aberrations.*')
warnings.filterwarnings('ignore', message='.*clipping to closest.*')

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from roman_disperser import elements, psf_model


def main():
    parser = argparse.ArgumentParser(
        description='Generate PSF caches for Roman grism disperser'
    )
    parser.add_argument(
        '--cache-dir', type=str, default='data/psf_cache',
        help='Cache directory (default: data/psf_cache)'
    )
    parser.add_argument(
        '--element', type=str, default='grism', choices=['grism', 'prism'],
        help='Dispersing element (default: grism)'
    )
    parser.add_argument(
        '--orders', type=str, default=None,
        help='Comma-separated orders to generate (default: the element '
             'orders with a distinct STPSF filter)'
    )
    parser.add_argument(
        '--detectors', type=str, default=None,
        help='Comma-separated detectors (default: all 18 WFI detectors)'
    )
    parser.add_argument(
        '--workers', '-j', type=int, default=1,
        help=f'Number of parallel workers (default: 1, max recommended: {os.cpu_count()})'
    )
    parser.add_argument(
        '--no-skip', action='store_true',
        help='Regenerate even if cache exists'
    )

    args = parser.parse_args()

    element = elements.get_element(args.element)

    # Parse orders; by default generate one cache per distinct STPSF filter
    # (grism order "2" reuses the order-1 cache, so it is not generated).
    if args.orders:
        orders = tuple(args.orders.split(','))
        unknown = [o for o in orders if o not in element.stpsf_filters]
        if unknown:
            parser.error(f"orders {unknown} not defined for {element.name}")
    else:
        seen = {}
        for o in element.orders:
            seen.setdefault(element.stpsf_filters[o], o)
        orders = tuple(seen.values())

    # PSF-grid wavelengths spanning the element band (matches the vendored
    # cache filenames; see elements.PSF_WL_STEP_UM)
    wavelengths = elements.psf_cache_wavelengths(element)

    # Parse detectors
    detectors = None
    if args.detectors:
        detectors = args.detectors.split(',')

    print("=" * 60)
    print("PSF Cache Generator")
    print("=" * 60)
    print(f"Cache directory: {args.cache_dir}")
    print(f"Element: {element.name}")
    print(f"Orders: {orders} "
          f"(filters {[element.stpsf_filters[o] for o in orders]})")
    print(f"Detectors: {detectors or 'all 18'}")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {not args.no_skip}")
    print("=" * 60)
    print()

    results = psf_model.generate_all_psf_caches(
        cache_dir=args.cache_dir,
        orders=orders,
        detectors=detectors,
        wavelengths=wavelengths,
        stpsf_filters=element.stpsf_filters,
        skip_existing=not args.no_skip,
        n_workers=args.workers,
        verbose=True,
    )

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if results['failed']:
        print("FAILURES:")
        for det, order, error in results['failed']:
            print(f"  {det} order {order}: {error}")
        sys.exit(1)
    else:
        print("All caches generated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
