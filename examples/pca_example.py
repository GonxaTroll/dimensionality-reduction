"""
Example: Basic PCA analysis with preprocessing and visualization using PCAModel.

This example demonstrates how to use the dimensionality_reduction package
via the PCAModel class for preprocessing, PCA, and visualization.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from dimensionality_reduction import PCAModel # pylint: disable=E0401


def main():
    """Run PCA example on Iris dataset using PCAModel."""
    # Load the Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.data # pylint: disable=E1101
    y = iris.target # pylint: disable=E1101

    print("=" * 60)
    print("BASIC PCA ANALYSIS - IRIS DATASET")
    print("=" * 60)
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Feature names: {list(df.columns)}\n")

    # Create and fit the PCA model
    print("Fitting PCA model with standardization...")
    model = PCAModel(df).fit(n_components=4, preprocessing="standardize")
    print(f"PCA fitted with {model.n_components} components")
    print(f"Explained variance ratio: {model.explained_variance_ratio_}")

    # Step 2: Visualize results
    print("\nCreating visualizations...")

    # Plot loadings for first component
    print("  - Plotting loadings (component 1)...")
    fig1 = model.plot_loadings(component_idx=0, figsize=(10, 6))
    plt.savefig("pca_loadings_component1.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("    Saved: pca_loadings_component1.png")

    # Plot data in PC space
    print("  - Plotting scores (PC1 vs PC2)...")
    fig2 = model.plot_scores(pc1=1, pc2=2, score_color=y, figsize=(10, 8))
    plt.savefig("pca_scores.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("    Saved: pca_scores.png")

    # Plot variance explained
    print("  - Plotting variance explained...")
    fig3 = model.plot_variance_explained(figsize=(10, 6))
    plt.savefig("pca_variance_explained.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("    Saved: pca_variance_explained.png")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_variance = sum(model.explained_variance_ratio_)
    total_variance_2_comp = sum(model.explained_variance_ratio_[:2])
    print(f"Total variance explained by 2 components: {total_variance_2_comp:.2%}")
    print(f"Total variance explained by all components: {total_variance:.2%}")
    print(f"\nPreprocessing method: {model.preprocessing}")
    print(f"Number of samples: {model.n_samples}")
    print(f"Number of features: {model.n_features}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
