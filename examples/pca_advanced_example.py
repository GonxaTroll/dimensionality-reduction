"""
Example: Advanced PCA analysis with outlier detection using PCAModel.

This example demonstrates comprehensive PCA analysis including visualization,
outlier detection, and diagnostics using T² and SPE statistics via PCAModel.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from dimensionality_reduction import PCAModel


def main():
    """Run comprehensive PCA analysis on Iris dataset using PCAModel."""
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    print("=" * 60)
    print("COMPREHENSIVE PCA ANALYSIS - IRIS DATASET")
    print("=" * 60)
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    print(f"Feature names: {list(df.columns)}\n")
    
    # Create and fit the PCA model
    print("=" * 60)
    print("STEP 1: MODEL FITTING")
    print("=" * 60)
    model = PCAModel(df).fit(n_components=4, preprocessing="standardize")
    print(f"PCA fitted with {model.n_components} components")
    print(f"Explained variance ratio: {model.explained_variance_ratio_}")
    print(f"Cumulative variance: {model.explained_variance_ratio_.cumsum()}")
    
    # Step 2: Component selection
    print("\n" + "=" * 60)
    print("STEP 2: COMPONENT SELECTION")
    print("=" * 60)
    kaiser_components = model.kaiser_components()
    selected_components = len(kaiser_components)
    print(f"Components with eigenvalue > 1: {selected_components}")
    
    # Plot variance explained
    fig1 = model.plot_variance_explained(figsize=(10, 6))
    plt.savefig("advanced_pca_variance_explained.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: advanced_pca_variance_explained.png")
    
    # Plot eigenvalues
    fig2 = model.plot_eigenvalues(figsize=(10, 6))
    plt.savefig("advanced_pca_eigenvalues.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: advanced_pca_eigenvalues.png")
    
    # Step 3: Score visualization
    print("\n" + "=" * 60)
    print("STEP 3: SCORE PLOTS")
    print("=" * 60)
    fig3 = model.plot_scores(pc1=1, pc2=2, score_color=y, figsize=(10, 8))
    plt.savefig("advanced_pca_scores.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: advanced_pca_scores.png")
    
    # Step 4: Loadings visualization
    print("\n" + "=" * 60)
    print("STEP 4: LOADINGS PLOTS")
    print("=" * 60)
    fig4 = model.plot_loadings_2d(pc1=1, pc2=2, draw_labels=True, 
                                   color_by="contrib", figsize=(10, 8))
    plt.savefig("advanced_pca_loadings_2d.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("Saved: advanced_pca_loadings_2d.png")
    
    # Step 5: Outlier detection - Hotelling T²
    print("\n" + "=" * 60)
    print("STEP 5: OUTLIER DETECTION - HOTELLING T²")
    print("=" * 60)
    t2_vals, t2_95, t2_99 = model.t2(selected_components, plot=True, figsize=(12, 6))
    plt.savefig("advanced_pca_t2_hotelling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_t2_hotelling.png")
    
    # Identify T² outliers
    t2_outliers, _ = model.outliers_from_t2(selected_components, alpha=0.99)
    if len(t2_outliers) > 0:
        print(f"T² outliers (99% confidence): {t2_outliers}")
    
    # Step 6: Outlier detection - SPE
    print("\n" + "=" * 60)
    print("STEP 6: OUTLIER DETECTION - SPE (Q-STATISTIC)")
    print("=" * 60)
    
    # SPE with chi-squared approximation
    spe_vals, spe_95, spe_99 = model.spe(selected_components, plot=True, figsize=(12, 6))
    plt.savefig("advanced_pca_spe.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_spe.png")
    
    # SPE with Jackson-Mudholkar approximation
    spe_jm_vals, spe_jm_95, spe_jm_99 = model.spe_jackson_mudholkar(
        selected_components, plot=True, figsize=(12, 6)
    )
    plt.savefig("advanced_pca_spe_jackson_mudholkar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_spe_jackson_mudholkar.png")
    
    # Identify SPE outliers
    spe_outliers, _ = model.outliers_from_spe(selected_components, alpha=0.99)
    if len(spe_outliers) > 0:
        print(f"SPE outliers (99% confidence): {spe_outliers}")
    
    # Step 7: Variable contributions for outliers
    print("\n" + "=" * 60)
    print("STEP 7: VARIABLE CONTRIBUTIONS FOR OUTLIERS")
    print("=" * 60)
    
    # Combine all outliers
    all_outliers = sorted(set(t2_outliers) | set(spe_outliers))
    
    if len(all_outliers) > 0:
        print(f"Analyzing variable contributions for samples: {all_outliers[:5]}")
        
        # Plot contributions for first few outliers
        for idx in all_outliers[:3]:
            fig, _ = model.plot_contribs(indivs=idx, pc=1, use_preprocessed=True, figsize=(12, 6))
            plt.savefig(f"advanced_pca_contributions_sample_{idx}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: advanced_pca_contributions_sample_{idx}.png")
    else:
        print("No significant outliers detected in this dataset.")
    
    # Step 8: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Components selected (Kaiser criterion): {selected_components}")
    print(f"Variance explained by selected components: "
          f"{model.explained_variance_ratio_[:selected_components].sum():.2%}")
    print(f"\nOutliers detected (99% confidence):")
    print(f"  - T²: {len(t2_outliers)} samples")
    print(f"  - SPE: {len(spe_outliers)} samples")
    print(f"  - Combined unique: {len(all_outliers)} samples")
    print(f"\nPreprocessing method: {model.preprocessing}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
