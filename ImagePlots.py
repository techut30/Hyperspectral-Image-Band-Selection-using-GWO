import numpy as np
import matplotlib.pyplot as plt



def plot_images(data, labels):
    # Define the folder path to save the images
    folder_path = './'

    # RGB Image using three selected bands (e.g., bands 29, 19, 9)
    rgb_image = np.zeros((data.shape[0], data.shape[1], 3))
    rgb_image[:, :, 0] = data[:, :, 29]  # Red channel
    rgb_image[:, :, 1] = data[:, :, 19]  # Green channel
    rgb_image[:, :, 2] = data[:, :, 9]   # Blue channel

    # Spectral Band Image (e.g., band 10) in grayscale
    spectral_band = data[:, :, 10]

    # Ground Truth labels
    ground_truth = labels

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # RGB image
    ax[0].imshow(rgb_image / rgb_image.max())  # Normalize for display
    ax[0].set_title("RGB Image")
    ax[0].axis('off')

    # Spectral band image
    ax[1].imshow(spectral_band, cmap='gray')
    ax[1].set_title("Spectral Band (Band 10)")
    ax[1].axis('off')

    # Ground truth labels
    ax[2].imshow(ground_truth, cmap='nipy_spectral')
    ax[2].set_title("Ground Truth")
    ax[2].axis('off')

    # Save the figure in the specified folder path
    fig.savefig(f"{folder_path}/indian_pine_visualization.png", dpi=300)
    plt.show()
