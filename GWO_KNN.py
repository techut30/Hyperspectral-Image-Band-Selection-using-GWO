import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import scipy.io as sio
import time
import threading
import os
import GrayWolfOpt
import CompositeImages as ci
import ImagePlots as ip

# Objective function for band selection
def band_selection_objective(selected_bands, data, labels):
    selected_bands = np.round(selected_bands).astype(int)
    selected_bands = np.unique(selected_bands)

    if len(selected_bands) < 1:
        return float("inf")

    X_selected = data[:, selected_bands]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=0.3, random_state=42)

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return 1 - accuracy_score(y_test, y_pred)  # Minimize error rate

# Load the dataset
def load_dataset():
    data_path = 'Indian_pines_corrected.mat'
    labels_path = 'Indian_pines_gt.mat'

    data = sio.loadmat(data_path)['indian_pines_corrected']
    labels = sio.loadmat(labels_path)['indian_pines_gt']

    # Reshape the data to be (pixels, bands)
    n_rows, n_cols, n_bands = data.shape
    data_reshaped = data.reshape((n_rows * n_cols, n_bands))
    labels_reshaped = labels.reshape((n_rows * n_cols,))

    # Remove pixels where labels are 0 (background/unlabeled)
    mask = labels_reshaped > 0
    data_reshaped = data_reshaped[mask]
    labels_reshaped = labels_reshaped[mask]

    return data_reshaped, labels_reshaped, data, labels


# Timer function
def timer(start_time, stop_event):
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"\rElapsed time: {elapsed_time:.2f} seconds", end="")
        time.sleep(1)  # Update every second

# Main execution
if __name__ == "__main__":
    data, labels, data_cube, labels_cube = load_dataset()

    ip.plot_images(data_cube, labels_cube)

    num_bands = data.shape[1]  # Now it's 2D, so the number of bands is the second dimension
    lb = 0  # lower bound of bands
    ub = num_bands - 1  # upper bound of bands
    dim = 30  # Number of bands to select

    stop_event = threading.Event()  # Event to signal the timer to stop
    start_time = time.time()  # Record the start time

    # Start the timer thread
    timer_thread = threading.Thread(target=timer, args=(start_time, stop_event))
    timer_thread.start()

    # Initialize the optimizer
    gwo = GrayWolfOpt.GrayWolfOptimizer(objective_function=lambda bands: band_selection_objective(bands, data, labels),
                            dim=dim, lb=lb, ub=ub, num_wolves=10, max_iter=50)

    # Perform optimization to select the best bands
    best_bands, best_score = gwo.optimize()

    stop_event.set()  # Signal the timer to stop
    timer_thread.join()  # Wait for the timer thread to finish

    # Print selected bands and best score
    print("\nBest bands selected:", np.round(best_bands).astype(int))
    print("Best score (error rate):", best_score)

    # Visualize the selected bands as a composite image
    output_directory = '.'  # Use the current directory
    unique_best_bands = np.unique(np.round(best_bands).astype(int))  # Ensure unique bands
    ci.visualize_selected_bands(data_cube, unique_best_bands, output_directory)