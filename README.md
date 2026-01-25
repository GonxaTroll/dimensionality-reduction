# Dimensionality Reduction Utils

A Python package containing utility functions for performing dimensionality reduction techniques over various datasets. This package provides comprehensive tools for data preprocessing, PCA analysis, visualization, and outlier detection.

## Features

- **Preprocessing utilities**: Functions for preparing data before applying dimensionality reduction
  - Standardization (zero mean, unit variance)
  - Normalization (min-max scaling)
  - Feature selection based on variance threshold
  
- **Visualization utilities**: Comprehensive tools for visualizing dimensionality reduction results
  - Plot PCA loadings (feature contributions) - both 1D and 2D
  - Visualize data in principal component space (score plots)
  - Plot explained variance ratios and elbow curves
  - Plot eigenvalues with Kaiser criterion
  - Variable contribution plots for specific observations
  
- **Analysis utilities**: Statistical methods for PCA diagnostics and outlier detection
  - Hotelling's T² statistic for score space outliers
  - SPE (Q-statistic) for residual space outliers
  - Jackson-Mudholkar approximation for SPE thresholds
  - Kaiser criterion for component selection
  - SCE (Squared prediction error) computation

## Installation

### From source

```bash
git clone https://github.com/GonxaTroll/dimensionality-reduction.git
cd dimensionality-reduction
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

### With examples

```bash
pip install -e ".[examples]"
```

## Quick Start

```python
import numpy as np
from sklearn.decomposition import PCA
from dimensionality_reduction.preprocessing import standardize
from dimensionality_reduction.visualization import (
    plot_loadings, plot_scores,
    compute_t2_hotelling, compute_spe, get_outlier_indexes
)

# Your data
X = np.random.randn(100, 10)

# Preprocess
X_standardized, mean, std = standardize(X)

# Apply PCA
pca = PCA(n_components=5)
X_transformed = pca.fit_transform(X_standardized)

# Visualize
plot_loadings(pca.components_.T, component_idx=0)
plot_scores(X_transformed, pc1=0, pc2=1)
plot_variance_explained(pca.explained_variance_ratio_, num_variables=10)

# Outlier detection
t2, t2_95, t2_99 = compute_t2_hotelling(
    X_transformed, pca.explained_variance_, selected_components=3
)
outliers = get_outlier_indexes(t2, t2_99)
print(f"Detected {len(outliers)} outliers")
```

## Documentation

### Preprocessing Module

#### `standardize(data, mean=None, std=None)`
Standardize data to have zero mean and unit variance.

**Parameters:**
- `data`: Input data array (n_samples, n_features)
- `mean`: Optional pre-computed mean values
- `std`: Optional pre-computed standard deviation values

**Returns:**
- Tuple of (standardized_data, mean, std)

#### `normalize(data, min_val=None, max_val=None)`
Normalize data to [0, 1] range.

**Parameters:**
- `data`: Input data array (n_samples, n_features)
- `min_val`: Optional pre-computed minimum values
- `max_val`: Optional pre-computed maximum values

**Returns:**
- Tuple of (normalized_data, min_val, max_val)

### Visualization Module

#### `plot_loadings(loadings, feature_names=None, component_idx=0, top_n=None, figsize=(10, 6))`
Plot loadings (feature contributions) for a principal component.

**Parameters:**
- `loadings`: Component loadings matrix (n_features, n_components)
- `feature_names`: Optional list of feature names
- `component_idx`: Index of component to plot
- `top_n`: If provided, only plot top N features by absolute loading value
- `figsize`: Figure size

**Returns:**
- Matplotlib figure object

#### `plot_scores(transformed_data, labels=None, pc1=0, pc2=1, figsize=(10, 8))`
Plot data in the space of two principal components.

**Parameters:**
- `transformed_data`: Transformed data (n_samples, n_components)
- `labels`: Optional labels for coloring points
- `pc1`: Component index for x-axis
- `pc2`: Component index for y-axis
- `figsize`: Figure size

**Returns:**
- Matplotlib figure object

#### `plot_variance_explained(explained_variance_ratio, figsize=(10, 6))`
Plot explained variance ratio for each component.

**Parameters:**
- `explained_variance_ratio`: Ratio of variance explained by each component
- `figsize`: Figure size

**Returns:**
- Matplotlib figure object

## Examples

See the `examples/` directory for complete usage examples:

```bash
python examples/pca_example.py
```

## Development

### Running tests

```bash
pytest tests/
```

### Running tests with coverage

```bash
pytest tests/ --cov=dimensionality_reduction --cov-report=term-missing
```

### Code formatting

```bash
black dimensionality_reduction/ tests/
```

### Linting

```bash
flake8 dimensionality_reduction/ tests/
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.3.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
