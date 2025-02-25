"""Main entry point for the Stella pipeline."""
import numpy as np
from utils.logger import setup_logger
from utils.config import get_config
from data_processing.data_loader import load_pose_data
from data_processing.feature_extraction import compute_velocity, compute_acceleration, compute_jerk
from self_supervised.training import train_simclr
from dimensionality_reduction.umap_reduction import reduce_dimensionality
from clustering.hdbscan_clustering import perform_hdbscan_clustering
from evaluation.cluster_evaluation import evaluate_clusters
from reporting.visualization import plot_clusters, plot_cluster_stability
from reporting.summary_generator import generate_cluster_summary

# Setup logger
logger = setup_logger()

def main():
    """Runs the full Stella pipeline."""
    config = get_config()
    logger.info("Loading data...")
    data = load_pose_data("data/pose_data.csv")
    
    logger.info("Extracting features...")
    velocity = compute_velocity(data)
    acceleration = compute_acceleration(velocity)
    jerk = compute_jerk(acceleration)
    features = np.concatenate([data, velocity, acceleration, jerk], axis=-1)
    
    logger.info("Training self-supervised model...")
    train_simclr(features, epochs=config["epochs"], batch_size=config["batch_size"])
    
    logger.info("Reducing dimensionality...")
    embedding, _ = reduce_dimensionality(features, config["umap_n_neighbors"], config["umap_min_dist"])
    
    logger.info("Performing clustering...")
    clusters, clusterer = perform_hdbscan_clustering(embedding, config["hdbscan_min_cluster_size"], config["hdbscan_min_samples"])
    
    logger.info("Evaluating clusters...")
    evaluation_metrics = evaluate_clusters(embedding, clusters)
    logger.info(f"Cluster Evaluation: {evaluation_metrics}")
    
    logger.info("Generating visualization and report...")
    plot_clusters(embedding, clusters)
    plot_cluster_stability(clusterer, clusters)
    cluster_summary = generate_cluster_summary(clusters)
    logger.info(f"Cluster Summary: {cluster_summary}")
    
if __name__ == "__main__":
    main()
