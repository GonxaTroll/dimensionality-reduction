# Dimensionality Reduction Utils

A Python package containing utility functions for performing dimensionality reduction techniques over various datasets. This package provides tools for data preprocessing and visualization of dimensionality reduction results.

## Features

- **Preprocessing utilities**: Functions for preparing data before applying dimensionality reduction
  - Standardization (zero mean, unit variance)
  - Normalization (min-max scaling)
  - Feature selection based on variance threshold
  
- **Visualization utilities**: Tools for visualizing dimensionality reduction results
  - Plot PCA loadings (feature contributions)
  - Visualize data in principal component space
  - Plot explained variance ratios

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

## Quick Start

```python
import numpy as np
from sklearn.decomposition import PCA
from dimensionality_reduction.preprocessing import standardize
from dimensionality_reduction.visualization import plot_loadings, plot_components

# Your data
X = np.random.randn(100, 10)

# Preprocess
X_standardized, mean, std = standardize(X)

# Apply PCA
pca = PCA(n_components=5)
X_transformed = pca.fit_transform(X_standardized)

# Visualize
plot_loadings(pca.components_.T, component_idx=0)
plot_components(X_transformed, component_x=0, component_y=1)
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

#### `select_features(data, variance_threshold=0.0)`
Select features based on variance threshold.

**Parameters:**
- `data`: Input data array (n_samples, n_features)
- `variance_threshold`: Features with variance below this will be removed

**Returns:**
- Tuple of (selected_data, feature_indices)

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

#### `plot_components(transformed_data, labels=None, component_x=0, component_y=1, figsize=(10, 8))`
Plot data in the space of two principal components.

**Parameters:**
- `transformed_data`: Transformed data (n_samples, n_components)
- `labels`: Optional labels for coloring points
- `component_x`: Component index for x-axis
- `component_y`: Component index for y-axis
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
