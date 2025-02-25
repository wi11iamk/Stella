# Stella: Self-Supervised Learning for Clinical Phenotyping of Pose Data

## Overview
Stella is a machine learning pipeline designed for **self-supervised phenotyping of upper limb motor impairments** using pose data collected in at-home settings. By leveraging **SimCLR, UMAP, and HDBSCAN**, it enables **unsupervised clustering** of movement patterns, providing clinicians with an automated tool for tracking stroke recovery and tailoring rehabilitation strategies.

### **Why At-Home Clinical Evaluations?**
Stroke is a leading cause of long-term disability worldwide, often affecting motor function in nuanced ways that traditional clinical assessments fail to capture. Socioeconomic barriers frequently limit access to in-clinic rehabilitation, making **remote, AI-driven motor function assessments** a crucial step toward democratizing stroke rehabilitation. 

**Stella** provides a scalable solution by using **pose estimation and machine learning** to analyze movement patterns, allowing patients to undergo detailed motor assessments **without frequent clinical visits**. By using a self-supervised learning framework, Stella reduces reliance on labeled datasets and provides **granular, patient-specific phenotyping** of motor impairments.

## Features
- **Self-Supervised Learning with 1D CNN + LSTM**: Adapts **SimCLR** to **time series data**, capturing both local and long-range dependencies in upper limb motion.
- **Dimensionality Reduction**: Applies **UMAP** to visualize high-dimensional movement features.
- **Unsupervised Clustering**: Uses **HDBSCAN** to identify distinct movement phenotypes.
- **Automated Reporting**: Generates clinician-friendly summaries and visualizations.
- **Configurable & Extendable**: Modular design allows easy adaptation to different datasets and rehabilitation protocols.

## Installation
```sh
# Clone the repository
git clone https://github.com/wi11iamk/Stella.git
cd Stella

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
stella/
├── data_processing/        # Load, preprocess, and extract motion features
│   ├── data_loader.py      # Loads pose data from files
│   ├── feature_extraction.py  # Computes velocity, acceleration, and jerk
│   ├── synthetic_data.py   # Generates structured synthetic pose data
├── self_supervised/        # Self-supervised contrastive learning (SimCLR)
│   ├── simclr.py           # SimCLR model with 1D CNN + LSTM
│   ├── augmentations.py    # Time-series data augmentations
│   ├── training.py         # Self-supervised model training loop
├── dimensionality_reduction/  # UMAP-based dimensionality reduction
│   ├── umap_reduction.py   # Applies UMAP for visualization
├── clustering/             # Clustering algorithms
│   ├── hdbscan_clustering.py  # HDBSCAN-based clustering
├── evaluation/             # Cluster evaluation metrics
│   ├── cluster_evaluation.py  # Silhouette Score, DBI, Dunn Index
├── reporting/              # Reporting and visualization tools
│   ├── visualization.py    # Cluster and stability visualization
│   ├── summary_generator.py  # Generates clinician-friendly reports
├── utils/                  # Configurations and logging
│   ├── config.py           # Global config settings
│   ├── logger.py           # Logging utility
├── tests/                  # Unit and integration tests
│   ├── test_data_processing.py
│   ├── test_self_supervised.py
│   ├── test_clustering.py
│   ├── test_evaluation.py
│   ├── test_reporting.py
│   ├── test_integration.py
├── main.py                 # Entry point for running the pipeline
└── README.md               # Project documentation
```

## Usage
Run the full pipeline:
```sh
python main.py
```

## Configuration
All hyperparameters (e.g., UMAP settings, batch size) are defined in `utils/config.py`.

## Logging
Execution logs are written to `stella.log`. Modify `utils/config.py` to change logging settings.

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
[MIT License](LICENSE)

## Author
**William D. Kistler** ([@wi11iamk](https://github.com/wi11iamk))
