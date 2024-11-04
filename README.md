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

```

### `GWO_RandomForest.py`

An alternative approach where band selection is performed using the **Random Forest classifier** instead of KNN. This script is structured similarly to main.py, with the following differences:

1. **Classifier**: Uses RandomForestClassifier with 100 estimators to evaluate the fitness of the selected bands.
2. **Parameters**: Adjusted parameters for GWO, including a higher number of bands and wolves.
3. **Visualization**: Visualizes the selected bands as a composite image.

### Usage

```python
python GWO_RandomForest.py
```

## Code Explanation 

### Objective Function for Band Selection

The objective function in both scripts takes in a set of selected bands, trains a classifier (KNN or Random Forest), and returns the error rate based on classification accuracy. The goal is to minimize this error rate, which represents the fitness score for GWO.

### DataSet Loading 

The dataset used is the Indian Pines hyperspectral dataset. The data is reshaped to (pixels, bands) format, and unlabeled pixels (background) are removed for accurate band selection.


### Gray Wolf Optimization 

The GWO algorithm optimizes the band selection process. The algorithm is configured with:

**dim**: Number of bands to select.

**lb and ub**: Lower and upper bounds for band indices.

**num_wolves and max_iter**: Parameters defining the number of wolves and iterations.

### Timer

A timer function displays the elapsed time for the optimization process.


## Example Outputs

Upon running each script, the output includes:

**Best Bands Selected**: The optimal subset of bands for classification.

**Best Score (Error Rate)**: The minimized classification error achieved.

**Composite Image**: A visualization of the selected bands.

## References 

1. Gray Wolf Optimizer: Inspired by the social hierarchy and hunting strategy of gray wolves.
2. Indian Pines Dataset: Commonly used hyperspectral dataset for classification tasks.





