# Data sources

The original animal datasets remain in their public repositories. This code distribution includes only compact model parameters and numerical summaries in `reference_data.zip`; the larger observations, predictions and trajectories are reconstructed locally.

## Beetles

**Jhawar, J. (2025). jiteshjhawar/Beetles_data_analysis: v1.0 — Beetles Data Analysis.**

- Dataset: [doi:10.5281/zenodo.17645611](https://doi.org/10.5281/zenodo.17645611).
- The Zenodo DOI record specifies [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- The [upstream v1.0 repository](https://github.com/jiteshjhawar/Beetles_data_analysis/tree/v1.0) additionally includes an MIT permission notice covering its code, data, and supplementary materials, with copyright **2025 Jitesh Jhawar**. The exact upstream notice is retained in `reference_data.zip` under `licenses`. These distinct notices are preserved rather than replaced by the analysis-code licence.

The preparation code transforms the published measurements into rotation observations and group features, reapplies the supplied fitted models, and reconstructs predictions and simulation trajectories. The observation and modelling procedures are described in the associated manuscript.

## Golden shiners

**Davidson, J. D.; Sosna, M. M. G.; Twomey, C. R.; Sridhar, V. H.; Leblanc, S. P.; Couzin, I. D. (2021). Collective detection based on visual information in animal groups.**

- Dataset: [doi:10.5061/dryad.sbcc2fr2h](https://doi.org/10.5061/dryad.sbcc2fr2h).
- Permission: [Creative Commons Zero 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
- Related article: [doi:10.1098/rsif.2021.0142](https://doi.org/10.1098/rsif.2021.0142).

The preparation code reads the published processed data, constructs positions and body headings for nine recordings in NPZ format, and rebuilds paired rotation observations and prediction results locally. Source DOI and checksum fields are preserved. Source arrays, original videos and HDF5 archives are not redistributed in this repository.

## Attribution and reuse

Cite the relevant dataset and study when using these materials. Their inclusion does not imply endorsement by the source creators. Source permissions apply to the corresponding source material; the MIT licence for this repository's analysis code does not relicense third-party data. Source checksums and download instructions are provided in SOURCE_DATA.md and the compact reference archive.
