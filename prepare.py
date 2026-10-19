"""Reconstruct large analysis inputs locally from the published source datasets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beetle-source', type=Path, help='Published beetle repository root or its data directory')
    parser.add_argument('--fish-source', type=Path, help='Published saved_data_and_results.zip or its data-grid.pklz')
    parser.add_argument('--data-dir', required=True, type=Path, help='Directory for locally reconstructed analysis inputs')
    parser.add_argument('--section', choices=('core', 'fish', 'all'), default='all')
    args = parser.parse_args()
    destination = args.data_dir.resolve()
    sources = [('core', args.beetle_source), ('fish', args.fish_source)]
    for section, source in sources:
        if args.section not in (section, 'all'):
            continue
        if source is None:
            parser.error(f'--{ "beetle" if section == "core" else "fish" }-source is required')
        source = source.resolve()
        if not source.exists():
            parser.error(f'Source path does not exist: {source}')
        if destination == source or (source.is_dir() and destination.is_relative_to(source)):
            parser.error('The reconstructed data directory must be outside the read-only source directory')
    if destination == Path(__file__).resolve().parent:
        parser.error('Choose a separate data directory, not the code directory itself')
    start = time.monotonic()
    from preparation.reference import reference_directory, install_reference
    results = {}
    with reference_directory() as reference:
        install_reference(reference, destination)
        if args.section in ('core', 'all'):
            from preparation.core import prepare_core
            results['core'] = prepare_core(args.beetle_source.resolve(), reference, destination)
        if args.section in ('fish', 'all'):
            from preparation.fish import prepare_fish
            results['fish'] = prepare_fish(args.fish_source.resolve(), reference, destination)
    result = {'passed': True, 'section': args.section, 'elapsed_seconds': time.monotonic() - start,
              'sections': results, 'sources_read_only': True,
              'scope': 'Locally reconstructed inputs from public source files and supplied model parameters.'}
    (destination / 'preparation_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: inputs prepared at {destination}', flush=True)
    return result


if __name__ == '__main__':
    main()
