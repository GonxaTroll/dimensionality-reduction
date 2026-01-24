# Examples

This directory contains example scripts demonstrating how to use the `dimensionality_reduction` package.

## Available Examples

### pca_example.py
A complete example showing:
- Data preprocessing with standardization
- Applying PCA using scikit-learn
- Visualizing PCA loadings
- Visualizing data in principal component space
- Plotting explained variance

**Requirements:**
```bash
pip install scikit-learn
```

**Usage:**
```bash
python examples/pca_example.py
```

This will create three visualization files:
- `pca_loadings_component1.png` - Feature contributions to PC1
- `pca_components.png` - Data plotted in PC1-PC2 space
- `pca_variance_explained.png` - Variance explained by each component
