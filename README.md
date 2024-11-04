# Hyperspectral-Image-Band-Selection-using-GWO

# Band Selection for Hyperspectral Image Analysis using Gray Wolf Optimization

This repository contains Python code for selecting the optimal bands from hyperspectral images using the **Gray Wolf Optimization (GWO)** algorithm. The GWO algorithm helps select the most informative spectral bands, enhancing the classification accuracy of hyperspectral images. Two methods are implemented for classification: **K-Nearest Neighbors (KNN)** and **Random Forest** classifiers.

## Overview

The repository includes two main files:
- `GWO_KNN.py`: Uses a K-Nearest Neighbors (KNN) classifier for band selection.
- `GWO_RandomForest.py`: Alternative approach that uses a Random Forest classifier for band selection.

Both files utilize Gray Wolf Optimization to identify the optimal subset of spectral bands for classification.

## Requirements

- Python 3.x
- NumPy
- Scikit-Learn
- Scipy
- Matplotlib
- `GrayWolfOpt` (Gray Wolf Optimization algorithm, should be implemented or imported)
- `CompositeImages` and `ImagePlots` (modules for visualization)

## Files

### `GWO_KNN.py`

This file performs band selection using the GWO algorithm and classifies using the **K-Nearest Neighbors** classifier.

1. **Objective Function**: Defines the objective function for GWO. The function minimizes the classification error rate based on selected spectral bands.
2. **Dataset Loading**: Loads the hyperspectral dataset (`Indian_pines_corrected.mat` for data and `Indian_pines_gt.mat` for labels).
3. **Optimization**: The GWO algorithm optimizes the selected bands to minimize the error rate.
4. **Visualization**: Selected bands are visualized as a composite image.

#### Usage
```python
python GWO_KNN.py
