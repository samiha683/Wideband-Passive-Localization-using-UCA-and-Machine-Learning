# Wideband Passive Localization using UCA and Machine Learning

Wideband-Passive-Localization-using-UCA-and-Machine-Learning implements a research / reproducible-experiments codebase for passive direction-of-arrival (DoA) and localization using a Uniform Circular Array (UCA) and machine learning. The project focuses on wideband signals (multi-frequency) received by an array of passive sensors and uses a mix of signal-processing feature extraction and supervised learning models to estimate source directions (azimuth / elevation) and/or 2D/3D locations.

This repository contains:
- a simulator / data generation pipeline for wideband UCA observations,
- preprocessing and feature-extraction code (TDOA, GCC-PHAT, phase differences, beamformed features),
- baseline classical algorithms (beamforming, MUSIC/Capon templates) and
- supervised ML baselines (SVM / RandomForest / MLP / CNN) with training and evaluation scripts.

TL;DR
- Generate or load data: python src/generate_data.py
- Train a model: python src/train.py --config configs/mlp.yaml
- Evaluate / infer: python src/evaluate.py --checkpoint models/mlp/best.pth

## Table of contents
- Background
- Features
- Results (summary)
- Repo layout
- Installation
- Quickstart
- Data generation
- Training
- Evaluation & inference
- Reproducing experiments
- Notes & limitations
- Contributing
- License & citation
- Contact

## Background

Passive localization using a UCA is attractive because a circular geometry provides uniform azimuthal resolution and enables compact arrays for 360° coverage. Wideband signals present both challenges and opportunities:
- Challenge: frequency-dependent phase and dispersion complicate narrowband DoA simplifications.
- Opportunity: wideband energy across multiple frequencies allows better time-delay estimation (TDE/TDOA) and richer features for learning-based estimators.

This repository investigates how classical signal processing (time-delay, beamforming) combined with learned models can improve robustness and accuracy in practical wideband, noisy conditions.

## Key features

- Wideband UCA data simulator with configurable:
  - number of sensors (array elements)
  - UCA radius
  - number of sources and SNR
  - signal bandwidth and frequency sampling
  - multipath / white Gaussian noise
- Feature extraction:
  - GCC-PHAT / cross-correlation based TDOA features
  - inter-element phase differences across frequency bins
  - frequency-domain snapshots and stacked spectrograms
  - steering-vector / beamformer outputs (coherent & incoherent)
- Baselines:
  - Classical: Delay-and-sum beamformer, MUSIC (narrowband approximation on sub-bands), Capon
  - Machine learning: SVM, RandomForest, MLP, simple 1D/2D CNNs
- End-to-end scripts for training, evaluation, plotting, and running inference
- Config-driven experiments (YAML) and save/load checkpoints

## Results (example / intended outcomes)

Results vary with array geometry, SNR and the difficulty of the scenario (number of simultaneous sources, multipath). Typical evaluation metrics included:
- Mean absolute error (degrees) for azimuth
- Root-mean-square localization error (meters) for 2D position
- Cumulative error distributions (CDF) and failure-rate at a threshold

Example findings reported in experiments (for reference only — actual numbers depend on your configuration and data):
- ML models trained on GCC-PHAT + phase features often outperform a single narrowband MUSIC run in low-to-moderate SNR for single-source scenarios.
- Combining beamformer outputs with learned fusion (MLP/CNN) increases resilience to SNR variations.

## Repository layout
- configs/                 - YAML experiment configs (models, dataset, training)
- data/                    - (gitignored) raw and processed datasets, splits
- docs/                    - notes, diagrams, references
- models/                  - saved model checkpoints produced by training
- notebooks/               - exploratory notebooks (visualization, debugging)
- results/                 - evaluation outputs (plots, metrics)
- src/
  - generate_data.py       - simulator & dataset export
  - features.py            - feature extraction utilities (GCC-PHAT, TDE, phase diffs)
  - models.py              - ML model definitions (MLP, CNN, classical wrappers)
  - train.py               - training harness
  - evaluate.py            - evaluation & metrics
  - predict.py             - single-sample inference
  - utils.py               - helpers (IO, plotting, seed handling)
- requirements.txt         - Python package requirements
- README.md                - you are reading this

## Installation

Minimum recommended:
- Python 3.8+
- 8+ GB RAM (more for larger datasets)
- Optional: NVIDIA GPU + CUDA for deep learning models

A quick setup (pip):
```bash
# create environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install requirements
pip install -r requirements.txt
```

requirements.txt (example)
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas
- h5py
- pyyaml
- tqdm
- librosa (optional, for extra audio utilities)
- torch (or tensorflow; repository supports PyTorch by default)
Adjust GPU/cu versions as appropriate (e.g., torch+cuda).

## Quickstart

1) Generate a small dataset for quick tests:
```bash
python src/generate_data.py --config configs/dev_generate.yaml --out data/dev/
```

2) Extract features (if separate step in your config):
```bash
python src/features.py --config configs/dev_features.yaml --in data/dev/ --out data/dev/features/
```

3) Train a baseline model (MLP):
```bash
python src/train.py --config configs/mlp.yaml
# model checkpoints saved to models/<experiment_name>/
```

4) Evaluate:
```bash
python src/evaluate.py --config configs/mlp.yaml --checkpoint models/mlp/best.pth --out results/mlp/
```

5) Predict (single sample):
```bash
python src/predict.py --checkpoint models/mlp/best.pth --input data/dev/sample_000.npz
```

## Data generation

The generator simulates wideband sources impinging on a UCA with optional additive noise and multipath. Configuration options include:
- array.num_elements
- array.radius_m
- signal.bandwidth_hz, center_frequency_hz
- dataset.num_samples, snr_db distribution
- environment.multipath = True/False

The generator saves each sample in a compact format (NumPy .npz or HDF5), including:
- raw_multichannel_waveforms or frequency-domain snapshots
- per-sample metadata: source azimuth/elevation/position, SNR, simulation parameters

## Training

Training is driven by YAML config files (see configs/). Typical keys:
- dataset: paths, batch_size, num_workers, shuffle, normalization
- model: type, architecture hyperparameters
- training: epochs, optimizer, learning_rate, lr_scheduler
- logging: logging_dir, checkpoint_freq, metric_to_monitor

Example train command:
```bash
python src/train.py --config configs/mlp.yaml
```

Training outputs:
- checkpoints (best + periodic)
- training logs (CSV or TensorBoard)
- training/validation metrics per epoch

## Evaluation & inference

Evaluation scripts compute:
- angular error statistics (mean, median, RMSE)
- localization error (when coordinates are estimated)
- confusion / sector accuracy if class-based
- plots (CDF, error histograms, predicted vs. ground truth scatter)

Run evaluation:
```bash
python src/evaluate.py --config configs/mlp.yaml --checkpoint models/mlp/best.pth --out results/mlp/
```

## Reproducing experiments

- Use the provided YAML configs in configs/ (example: configs/experiments/).
- Fix random seeds in configs for determinism (note: GPU nondeterminism may still exist).
- Save config copies alongside checkpoints so experiments are reproducible.
- Use the included notebooks to visualize dataset and intermediate features.

## Notes & limitations

- Simulated data may not fully capture real-world propagation impairments (hardware imperfections, correlated noise, strong multipath).
- The code uses sub-band narrowband approximations for some classical algorithms (e.g., MUSIC). For strictly correct wideband MUSIC you should implement coherent signal-subspace combining or use wideband-specific steering models.
- For large-scale or real-time deployments, this repository is primarily a research / prototyping baseline. Profiling and optimization will be required.

## Contributing

Contributions are welcome. Suggested workflow:
- Fork the repo and create a feature branch.
- Add tests or example notebooks for new functionality.
- Open a PR describing your change and the motivation.
Please follow the code style used in the repo and include a short unit test or example where appropriate.

## License & citation

- License: MIT (see LICENSE file)
If you use this work in academic research, please cite the repo and the corresponding paper (if applicable). A suggested citation:
> Samiha683 et al., "Wideband Passive Localization using UCA and Machine Learning", GitHub repository, 2026.

## Contact

Maintainer: samiha683 (GitHub)
For questions or issues, please open an issue on the repository.

## Acknowledgements & references

- Uniform Circular Array (UCA) literature and wideband DOA references (e.g., D. Johnson & D. Dudgeon, "Array Signal Processing", and other standard textbooks)
- GCC-PHAT and TDOA methods
- MUSIC and Capon beamforming techniques

## Troubleshooting

- If training takes too long: reduce dataset size, lower epochs, or use the dev configs.
- If results are unstable: check normalization of inputs and that training/validation splits do not overlap.

## Appendix: Example configs

A minimal example (configs/mlp.yaml):
```yaml
dataset:
  path: data/dev/features/
  batch_size: 64
  num_workers: 4

model:
  type: mlp
  hidden_sizes: [256, 128]
  dropout: 0.3
  output: regression  # "regression" for continuous azimuth/position, "classification" for sector labels

training:
  epochs: 200
  optimizer: adam
  lr: 1e-3
  weight_decay: 1e-5
  seed: 42

logging:
  log_dir: runs/mlp_experiment
  checkpoint_dir: models/mlp
```
