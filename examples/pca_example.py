"""
Example: Basic PCA analysis with preprocessing and visualization.

This example demonstrates how to use the dimensionality_reduction package
to preprocess data, perform PCA, and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Import dimensionality_reduction utilities
from dimensionality_reduction.preprocessing import standardize
from dimensionality_reduction.visualization import (
    plot_loadings, plot_components, plot_variance_explained
)


def main():
    """Run PCA example on Iris dataset."""
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    print("Original data shape:", X.shape)
    print("Feature names:", feature_names)
    
    # Step 1: Preprocess the data (standardization)
    print("\n1. Standardizing the data...")
    X_standardized, mean, std = standardize(X)
    print(f"Mean of standardized data: {np.mean(X_standardized, axis=0)}")
    print(f"Std of standardized data: {np.std(X_standardized, axis=0)}")
    
    # Step 2: Apply PCA
    print("\n2. Applying PCA...")
    pca = PCA(n_components=4)
    X_transformed = pca.fit_transform(X_standardized)
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Step 3: Visualize results
    print("\n3. Creating visualizations...")
    
    # Plot loadings for first component
    print("   - Plotting loadings...")
    fig1 = plot_loadings(
        pca.components_.T,
        feature_names=feature_names,
        component_idx=0,
        figsize=(10, 6)
    )
    plt.savefig("pca_loadings_component1.png", dpi=150, bbox_inches='tight')
    print("     Saved: pca_loadings_component1.png")
    
    # Plot data in PC space
    print("   - Plotting components...")
    fig2 = plot_components(
        X_transformed,
        labels=y,
        component_x=0,
        component_y=1,
        figsize=(10, 8)
    )
    plt.savefig("pca_components.png", dpi=150, bbox_inches='tight')
    print("     Saved: pca_components.png")
    
    # Plot variance explained
    print("   - Plotting variance explained...")
    fig3 = plot_variance_explained(pca.explained_variance_ratio_)
    plt.savefig("pca_variance_explained.png", dpi=150, bbox_inches='tight')
    print("     Saved: pca_variance_explained.png")
    
    # Print summary
    print("\n4. Summary:")
    print(f"   Total variance explained by 2 components: "
          f"{sum(pca.explained_variance_ratio_[:2]):.2%}")
    print(f"   Total variance explained by all components: "
          f"{sum(pca.explained_variance_ratio_):.2%}")
    
    plt.close('all')
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
