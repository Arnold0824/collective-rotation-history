# Collective Rotation History

Reproducible analyses for **Recent history improves prediction of collective rotation in beetle and fish groups**.

This repository provides analysis code, fitted model parameters and compact numerical summaries. Source animal data remain in their public repositories. Large observations, prediction tables and simulated trajectories are reconstructed locally; no separate large data release from this project is required.

## Install

Use Python 3.11 or 3.12. No GPU is required. Install the dependencies and run the analysis scripts from the cloned repository.

```bash
git clone https://github.com/Arnold0824/collective-rotation-history.git
cd collective-rotation-history
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```bash
python -m pip install -r requirements-tested.txt
```

The tested Windows environment uses Python 3.11.7, NumPy 1.26.4, pandas 2.2.2, PyArrow 14.0.2, SciPy 1.13.1, Matplotlib 3.9.1.post1 and PyMuPDF 1.28.2. `requirements.txt` also lists supported version ranges; not every combination has been tested.

## Generate the figures

The included `reference_data.zip` contains the small numerical summaries needed for all 7 main figures and 10 supplementary figures:

```bash
python reproduce.py --section figures --output-dir ./outputs
```

Figures are written in PDF, SVG and PNG formats. This command renders saved results; it does not fit models or reconstruct the experiments.

## Reconstruct and check the analyses

Obtain the public source data described in [SOURCE_DATA.md](SOURCE_DATA.md):

- **Beetles:** the data files from [Zenodo v1.0](https://doi.org/10.5281/zenodo.17645611), also linked to the [upstream v1.0 repository](https://github.com/jiteshjhawar/Beetles_data_analysis/tree/v1.0).
- **Golden shiners:** `saved_data_and_results.zip` from [Dryad](https://doi.org/10.5061/dryad.sbcc2fr2h), approximately 1.05 GB. The individual video archives are not needed for this workflow.

Download the 30 required beetle files directly from their original host with checksum verification (approximately 1.17 GB):

```bash
python fetch_sources.py --destination ./sources
```

Existing verified beetle files are reused. Download the fish archive through the Dryad dataset webpage and save it as `./sources/fish/saved_data_and_results.zip`; the automated Dryad API currently requires authentication. The preparation step verifies its size and checksum. Store sources separately from generated data and outputs, then construct the analysis inputs:

```bash
python prepare.py --beetle-source ./sources/beetles --fish-source ./sources/fish/saved_data_and_results.zip --data-dir ./work/data
```

If you already downloaded the files, use their actual paths instead of the example source paths above. The beetle argument accepts the upstream repository root or its data directory; the fish argument also accepts an extracted `data-grid.pklz`. Preparation reads the source files without changing them, applies the recorded observation rules, and reconstructs the large inputs locally. Fish preparation repeats model selection and fitting; beetle predictions and simulated trajectories use the supplied fitted parameters and simulation settings.

Run the independent saved-reference comparisons:

```bash
python reproduce.py --data-dir ./work/data --output-dir ./outputs --section all
```

Both commands write a JSON summary and print `PASS` on success. Large files in `work/` and generated outputs are excluded from Git.

## Computational scope

| Section | Calculation |
| --- | --- |
| `prepare.py --section core` | Construct beetle observations, group features and displacement sensitivities; reapply saved predictors and state-age models; regenerate trajectories using the recorded settings. Requires `--beetle-source`. |
| `prepare.py --section fish` | Recover nine source arrays and paired directional observables, then fit the complete paired fish analysis. Requires `--fish-source`. |
| `reproduce.py --section core` | Recompute forecast and state-age scores, replay model parameters, and reconstruct reversal events, observation times and supported waiting comparisons. |
| `reproduce.py --section fish` | Check source observations, common samples, time alignment, training-only selection and prediction scores; independently refit 20 selected models. |
| `reproduce.py --section figures` | Generate all 17 figures from the numerical summaries. Does not require the original source files. |

`all` is the default section for both commands. `reproduce.py --section fish --refit-fish-all` can additionally repeat the complete paired fish fit after preparation. Each complete fish fit covers 810 continuous models and 108 categorical models. The workflow does not retrain all beetle models or repeat tracking from video.

## Design and outputs

Beetle development uses recording D at each group size (50, 100 and 200); evaluation uses A, B, C and E. The seven predictors use common evaluation samples at horizons of 3, 10 and 30 frames. Fish comparisons use three recordings at each of 10, 30 and 70 fish, with two training recordings and one held-out recording per fold. Body-heading and movement-based rotation use matched samples and respect tracking-identity boundaries.

Unfavorable results remain included. The directional comparisons reuse the same fish recordings and do not add biological replication. Predictive gains do not establish a unique physical memory mechanism or accurate reproduction of all generated statistics.

Main output files are `preparation_summary.json`, `reproduction_summary.json`, `core/numerical_verification.json`, `fish/verification.json`, and `figures/figure_manifest.json`. Figure S9 is the paired fish-direction comparison; Figure S10 contains the temporal and spectral comparisons.

## Files and citation

`prepare.py` constructs local inputs; `reproduce.py` runs scores and figures. `preparation/`, `methods/`, `fish/` and `figures/` contain the corresponding implementations. `reference_data.zip` stores model parameters, reference scores and compact figure inputs. It contains no original videos, source fish arrays or large prediction tables.

Cite the associated study and source datasets when using these analyses. Dataset citations and permissions are in [DATA_SOURCES.md](DATA_SOURCES.md); `CITATION.cff` identifies the software. The analysis code uses the [MIT License](LICENSE). Original datasets retain their own permissions and attribution requirements.
