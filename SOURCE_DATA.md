# Obtaining the source data

Download these datasets from their original repositories. `SOURCE_FILES.csv` lists all 31 required files, direct URLs, byte sizes and SHA-256 checksums. The optional `fetch_sources.py` helper downloads and verifies the 30 beetle files. Obtain the fish archive through the Dryad webpage; its automated download API currently requires authentication. They are not redistributed with this code. Data preparation writes large derived files locally and leaves the downloaded source files unchanged.

## Beetles

- Dataset: [Jhawar (2025), Beetles Data Analysis, v1.0](https://doi.org/10.5281/zenodo.17645611).
- Fixed upstream version: [Beetles_data_analysis v1.0](https://github.com/jiteshjhawar/Beetles_data_analysis/tree/v1.0).
- Give `prepare.py --beetle-source` the repository root or its data directory. The preparation code verifies the required source files against recorded checksums before use.
- Obtain the actual data files if the repository archive contains Git LFS pointer files. Pointer files are not numerical data and will fail the checksum check. The preparation source manifest identifies the required files.

## Golden shiners

- Dataset: [Davidson et al. (2021), Collective detection based on visual information in animal groups](https://doi.org/10.5061/dryad.sbcc2fr2h).
- Required archive: `saved_data_and_results.zip`, selected on the dataset webpage above.
- Exact archive size: 1,053,800,358 bytes.
- Archive SHA-256: `e2bde9af7d112d6f0abe75808682d4ca7a86dd0515bed54f5877a8512cd00dbd`.
- Required member: `savedresults-final/data-grid.pklz` (903,184,076 bytes).
- Member SHA-256: `cc2d4131ad8e3825c84885aee4f2c8f5cb5ce3934b0ce604dcd68a9f1becd49a`.

The nine analysis recordings are 0066, 0105 and 0126 at 10 fish; 0084, 0115 and 0120 at 30 fish; and 0103, 0107 and 0124 at 70 fish. The required processed positions and headings are in the single archive above. The much larger individual video/HDF5 archives are not required.

Pass either the downloaded ZIP or its extracted `data-grid.pklz` to `prepare.py --fish-source`. The checksum-verified source is converted into the analysis arrays, after which the paired observation definitions and model fits are reconstructed.

## What this repository supplies

The compact `reference_data.zip` contains this study's model parameters, analysis settings and numerical result summaries. These files are different from the public animal recordings. Large prediction tables and generated trajectories are computed on the user's machine and are not needed as GitHub uploads.
