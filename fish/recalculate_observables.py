"""Independent vectorized reconstruction from published per-fish arrays."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .paths import data_root, fish_input, fish_output
ROOT = data_root()
OUT = fish_output()
OUT.mkdir(parents=True,exist_ok=True)
SOURCE = ROOT / 'data/fish_directions'


def rotation(points, directions, eligible):
    count = eligible.sum(axis=1)
    center = np.where(eligible[..., None], points, 0).sum(axis=1) / np.maximum(count[:, None], 1)
    radial = points - center[:, None, :]
    rn = np.linalg.norm(radial, axis=2)
    dn = np.linalg.norm(directions, axis=2)
    good = eligible & (rn > 1e-12) & (dn > 1e-12)
    cross = radial[..., 0] * directions[..., 1] - radial[..., 1] * directions[..., 0]
    contribution = np.divide(cross, rn * dn, out=np.zeros_like(cross), where=good)
    value = contribution.sum(axis=1) / np.maximum(good.sum(axis=1), 1)
    value[(count / points.shape[1] < .8) | (good.sum(axis=1) == 0)] = np.nan
    return np.clip(value, -1, 1)


def main():
    checks, summaries, boundary_records = [], [], []
    folder = OUT / 'reconstructed'
    folder.mkdir(exist_ok=True)
    for path in sorted(SOURCE.glob('N*.npz')):
        with np.load(path, allow_pickle=False) as z:
            n = int(z['group_size'])
            if n == 151:
                continue
            trial = str(z['trial'].item())
            points = z['positions']; angles = z['heading_angles']; frames = z['frames']
            stored = z['milling']
            direction = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            finite = np.isfinite(points).all(axis=2) & np.isfinite(angles)
            original = rotation(points, direction, finite)
            difference = np.abs(original - stored)
            maximum = float(np.nanmax(difference))
            checks.append(dict(trial=f'N{n}_{trial}', original_milling_max_abs_difference=maximum,
                               finite_mask_equal=bool(np.array_equal(np.isfinite(original), np.isfinite(stored))),
                               frames_consecutive=bool(np.all(np.diff(frames) == 1))))
            delta = np.full_like(points, np.nan)
            adjacent = np.diff(frames) == 1
            delta[1:] = np.where(adjacent[:, None, None], points[1:] - points[:-1], np.nan)
            length = np.linalg.norm(delta, axis=2)
            candidates = np.flatnonzero(np.nanmedian(length, axis=1) > 50)
            cuts = []
            for index in candidates:
                cost = np.linalg.norm(points[index, :, None, :] - points[index-1, None, :, :], axis=2)
                row, col = linear_sum_assignment(cost)
                matched_median = float(np.median(cost[row, col]))
                if matched_median < 10:
                    cuts.append(int(index))
                    boundary_records.append(dict(group_size=n, trial=trial, frame=int(frames[index]),
                        index_displacement_median=float(np.median(length[index])),
                        assigned_displacement_median=matched_median,
                        assigned_changed_fraction=float(np.mean(row != col))))
            assert cuts == [3596,7191,10787,14382,17977,21573], (n,trial,cuts)
            delta[cuts] = np.nan
            length[cuts] = np.nan
            for threshold in (1e-12, .5, 1.):
                common = finite & np.isfinite(delta).all(axis=2) & (length > threshold)
                mh = rotation(points, direction, common)
                md = rotation(points, delta, common)
                valid = np.isfinite(mh) & np.isfinite(md)
                cosine = np.divide((direction * delta).sum(axis=2), length,
                                   out=np.full_like(length, np.nan), where=common)
                correlation = float(np.corrcoef(mh[valid], md[valid])[0, 1]) if valid.sum() > 1 else None
                summaries.append(dict(group_size=n, trial=trial, threshold=threshold,
                    frames=len(frames), valid_frames=int(valid.sum()),
                    fraction=float(valid.mean()), correlation=correlation,
                    mean_absolute_difference=float(np.mean(np.abs(mh[valid]-md[valid]))),
                    heading_movement_cosine_mean=float(np.nanmean(cosine)),
                    positive_direction_fraction=float(np.nanmean(cosine[common] > 0)),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                if threshold == 1e-12:
                    pd.DataFrame(dict(frame=frames, original_heading=original,
                        heading_common=mh, displacement_common=md,
                        common_individual_count=common.sum(axis=1))).to_parquet(
                            folder/f'N{n}_{trial}.parquet', index=False)
        print(f'N{n}_{trial}: original maximum difference {maximum:.3g}', flush=True)
    assert len(checks) == 9
    assert all(c['original_milling_max_abs_difference'] < 1e-12 and c['finite_mask_equal']
               and c['frames_consecutive'] for c in checks)
    (OUT/'observable_reconstruction_checks.json').write_text(json.dumps(checks, indent=2))
    pd.DataFrame(summaries).to_csv(OUT/'observation_definition_quality.csv', index=False)
    pd.DataFrame(boundary_records).to_csv(OUT/'independent_identity_boundaries.csv', index=False)
    print('Nine source observables reconstructed successfully.')


if __name__ == '__main__':
    main()
