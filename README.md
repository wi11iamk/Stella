# Stella: Self-Supervised Learning for Clinical Phenotyping of Pose Data

## Overview
Stella is a machine learning pipeline designed for **self-supervised phenotyping of upper limb motor impairment** using pose data collected from the comfort of home. By leveraging **SimCLR, UMAP, and HDBSCAN**, it enables **unsupervised clustering** of movement patterns, providing clinicians with an automated tool for tracking motor recovery and tailoring rehabilitation strategies.

### **Why At-Home Clinical Evaluations?**
Stroke is a leading cause of long-term disability worldwide, often having a significant impact on motor function. Socioeconomic barriers, as well as US and UK therapy quotas, limit access to in-clinic rehabilitation, making **remote, ML-driven motor function assessments** a step toward democratizing and expanding rehabilitation. 

**Stella** provides a scalable solution by using **pose estimation and machine learning** to analyse movement patterns of the upper limbs, allowing patients to undergo detailed motor assessments **without frequent clinical visits**. By using a self-supervised learning framework, Stella reduces reliance on labeled datasets and provides granular phenotyping of motor impairment.

## Features
- **Self-Supervised Learning with 1D CNN + LSTM**: Adapts **SimCLR** to **time series data**, capturing both local and long-range dependencies in upper limb motion.
- **Dimensionality Reduction**: Applies **UMAP** to visualize high-dimensional movement features.
- **Unsupervised Clustering**: Uses **HDBSCAN** to identify distinct movement phenotypes.
- **Automated Reporting**: Generates clinician-friendly summaries and visualizations.
- **Configurable & Extendable**: Modular design allows easy adaptation to different pose datasets.

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
├── data_processing/           # Load, preprocess, and extract motion features
├── self_supervised/           # Self-supervised contrastive learning (SimCLR with 1D CNN + LSTM)
├── dimensionality_reduction/  # UMAP-based dimensionality reduction
├── clustering/                # Clustering algorithms
├── evaluation/                # Cluster evaluation with Silhouette Score, DBI, Dunn Index
├── reporting/                 # Reporting and visualization tools
├── utils/                     # Configurations and logging
├── main.py
└── README.md
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

## **License**
This project is licensed under the MIT License.

## Contributions
If you're interested in improving Stella, feel free to submit a pull request or open an issue.

**Maintainers**: `@wi11iamk`

## **Acknowledgments**
Special thanks to the researchers and clinicians in rehabilitation science @UCLH National Hospital for Neurology and Neurosurgery (NHNN), Queen Square for inspiring and contributing to this project.

