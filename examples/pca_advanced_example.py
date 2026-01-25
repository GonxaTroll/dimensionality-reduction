"""
Example: Advanced PCA analysis with outlier detection.

This example demonstrates how to use the enhanced dimensionality_reduction package
for comprehensive PCA analysis including visualization, outlier detection, and
diagnostics using T² and SPE statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Import dimensionality_reduction utilities
from dimensionality_reduction.preprocessing import standardize
from dimensionality_reduction.visualization import (
    plot_variance_explained,
    plot_eigenvalues, plot_scores, plot_loadings_2d,
    plot_contribs, get_last_best_eigenvalue, compute_t2_hotelling,
    compute_spe, compute_spe_jackson_mudholkar, get_outlier_indexes
)


def main():
    """Run comprehensive PCA analysis on Iris dataset."""
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    print("=" * 60)
    print("COMPREHENSIVE PCA ANALYSIS - IRIS DATASET")
    print("=" * 60)
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    
    # Step 1: Preprocess the data
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    X_standardized, mean, std = standardize(X)
    print("Data standardized successfully")
    
    # Step 2: Apply PCA
    print("\n" + "=" * 60)
    print("STEP 2: PCA TRANSFORMATION")
    print("=" * 60)
    pca = PCA(n_components=4)
    X_transformed = pca.fit_transform(X_standardized)
    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Step 3: Component selection
    print("\n" + "=" * 60)
    print("STEP 3: COMPONENT SELECTION")
    print("=" * 60)
    
    # Using Kaiser criterion
    best_eigenvalues = get_last_best_eigenvalue(pca.explained_variance_)
    selected_components = len(best_eigenvalues)
    
    # Plot elbow curve
    fig1 = plot_variance_explained(
        pca.explained_variance_ratio_,
        num_variables=X.shape[1],
        figsize=(10, 6)
    )
    plt.savefig("advanced_pca_elbow.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: advanced_pca_elbow.png")
    
    # Plot eigenvalues
    fig2 = plot_eigenvalues(pca.explained_variance_, figsize=(10, 6))
    plt.savefig("advanced_pca_eigenvalues.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: advanced_pca_eigenvalues.png")
    
    # Step 4: Score visualization
    print("\n" + "=" * 60)
    print("STEP 4: SCORE PLOTS")
    print("=" * 60)
    
    # Score plot with species coloring
    fig3 = plot_scores(
        X_transformed, 
        pc1=1, 
        pc2=2, 
        score_color=y,
        figsize=(10, 8)
    )
    plt.savefig("advanced_pca_scores.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: advanced_pca_scores.png")
    
    # Step 5: Loadings visualization
    print("\n" + "=" * 60)
    print("STEP 5: LOADINGS PLOTS")
    print("=" * 60)
    
    # 2D loadings plot with contributions
    fig4 = plot_loadings_2d(
        pca.components_, 
        variables=feature_names,
        pc1=1, 
        pc2=2,
        draw_labels=True,
        color_by="contrib",
        figsize=(10, 8)
    )
    plt.savefig("advanced_pca_loadings_2d.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("Saved: advanced_pca_loadings_2d.png")
    
    # Step 6: Outlier detection - Hotelling T²
    print("\n" + "=" * 60)
    print("STEP 6: OUTLIER DETECTION - HOTELLING T²")
    print("=" * 60)
    
    t2, t2_95, t2_99 = compute_t2_hotelling(
        X_transformed,
        pca.explained_variance_,
        selected_components,
        plot=True,
        figsize=(12, 6)
    )
    plt.savefig("advanced_pca_t2_hotelling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_t2_hotelling.png")
    
    # Identify T² outliers
    t2_outliers_95 = get_outlier_indexes(t2, t2_95)
    t2_outliers_99 = get_outlier_indexes(t2, t2_99)
    
    if len(t2_outliers_99) > 0:
        print(f"\nT² outliers (99% confidence): {t2_outliers_99}")
    
    # Step 7: Outlier detection - SPE (Q-statistic)
    print("\n" + "=" * 60)
    print("STEP 7: OUTLIER DETECTION - SPE (Q-STATISTIC)")
    print("=" * 60)
    
    # SPE with chi-squared approximation
    spe, spe_95, spe_99 = compute_spe(
        X_standardized,
        X_transformed,
        pca.components_,
        selected_components,
        plot=True,
        figsize=(12, 6)
    )
    plt.savefig("advanced_pca_spe.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_spe.png")
    
    # SPE with Jackson-Mudholkar approximation
    spe_jm, spe_jm_95, spe_jm_99 = compute_spe_jackson_mudholkar(
        X_standardized,
        X_transformed,
        pca.components_,
        pca.explained_variance_,
        selected_components,
        plot=True,
        figsize=(12, 6)
    )
    plt.savefig("advanced_pca_spe_jackson_mudholkar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_spe_jackson_mudholkar.png")
    
    # Identify SPE outliers
    spe_outliers_99 = get_outlier_indexes(spe, spe_99)
    
    if len(spe_outliers_99) > 0:
        print(f"\nSPE outliers (99% confidence): {spe_outliers_99}")
    
    # Step 8: Variable contributions for outliers
    print("\n" + "=" * 60)
    print("STEP 8: VARIABLE CONTRIBUTIONS FOR OUTLIERS")
    print("=" * 60)
    
    # Combine all outliers
    all_outliers = np.unique(np.concatenate([t2_outliers_99, spe_outliers_99]))
    
    if len(all_outliers) > 0:
        print(f"Analyzing variable contributions for samples: {all_outliers[:5]}")
        
        # Plot contributions for first few outliers
        for idx in all_outliers[:3]:  # Analyze first 3 outliers
            fig, contribs = plot_contribs(
                X_standardized,
                pca.components_,
                indivs=int(idx),
                pc=1,
                variable_names=feature_names,
                simca_style=True,
                figsize=(12, 6)
            )
            plt.savefig(f"advanced_pca_contributions_sample_{idx}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: advanced_pca_contributions_sample_{idx}.png")
    else:
        print("No significant outliers detected in this dataset.")
    
    # Step 9: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Components selected (Kaiser criterion): {selected_components}")
    print(f"Variance explained by selected components: "
          f"{sum(pca.explained_variance_ratio_[:selected_components]):.2%}")
    print(f"\nOutliers detected:")
    print(f"  - T² (99% confidence): {len(t2_outliers_99)} samples")
    print(f"  - SPE (99% confidence): {len(spe_outliers_99)} samples")
    print(f"  - Combined unique outliers: {len(all_outliers)} samples")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
