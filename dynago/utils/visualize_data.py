import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize():
    # Load CSV: assume last column is the gesture id
    data = pd.read_csv("dynago/data/normalized.csv")
    X = data.iloc[:, :-1].values  # all landmark columns
    y = data.iloc[:, -1].values  # gesture id column

    # Reduce dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Landmark Dataset Visualization")
    plt.colorbar(label="Gesture ID")
    plt.savefig(f"dynago/performance/dataset_visualization.png", dpi=300)
    plt.show()
