import numpy as np
import matplotlib.pyplot as plt
import os

# Function to visualize selected bands as a composite image
def visualize_selected_bands(data_cube, selected_bands, output_dir):
    # Reshape data cube back to original spatial dimensions
    n_rows, n_cols, _ = data_cube.shape

    # Select three bands from the selected bands
    if len(selected_bands) >= 3:
        red_band, green_band, blue_band = selected_bands[:3]
    else:
        raise ValueError("Need at least 3 bands to create a composite image.")

    # Create a composite image
    composite_image = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    composite_image[:, :, 0] = data_cube[:, :, red_band]  # Red channel
    composite_image[:, :, 1] = data_cube[:, :, green_band]  # Green channel
    composite_image[:, :, 2] = data_cube[:, :, blue_band]  # Blue channel

    # Normalize the image for better visualization
    composite_image -= composite_image.min()
    composite_image /= composite_image.max()

    # Save the composite image
    output_path = os.path.join(output_dir, "selected_bands_composite.png")
    plt.imshow(composite_image)
    plt.title("Composite Image of Selected Bands")
    plt.axis("off")
    plt.savefig(output_path)
    print(f"Composite image saved at: {output_path}")
    plt.show()
