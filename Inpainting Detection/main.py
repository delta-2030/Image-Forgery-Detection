import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_image(title, image, cmap="gray"):               # Displaying the image
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()
def load_image(path, to_grayscale=False):
    image = Image.open(path)
    if to_grayscale:
        image = image.convert("L") 
    return np.array(image)
def pixel_difference_analysis(original, suspected):                 # Function to calculate and display pixel-wise differences between original and suspected image
    diff = np.zeros_like(original, dtype=np.uint8)                  # Initializing the an array for storing differences
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            diff_value = abs(int(original[i, j]) - int(suspected[i, j]))  
            diff[i, j] = np.clip(diff_value, 0, 255)                 # Clipping the values to stay within the valid range
    show_image("Pixel-Level Difference", diff)
    return diff
def manual_convolution(image, kernel):                               # Function to apply a convolution filter manually on the image
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2                # Defining the Pad size
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')        # Padding the image by mirroring its border values symmetrically
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(region * kernel) 
    return result
def edge_detection(image):                                         # Function to perform edge detection using the Sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])        # Sobel Kernel for the X direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])        # Sobel Kernels for the Y direction
    gradient_x = manual_convolution(image, sobel_x)
    gradient_y = manual_convolution(image, sobel_y)
    edges = np.sqrt(gradient_x**2 + gradient_y**2)
    edges = (edges / edges.max()) * 255  
    edges = edges.astype(np.uint8)
    return edges
def manual_median_filter(image, size=5):                           # Function to apply a manual median filter for noise reduction.
    height, width = image.shape
    pad = size // 2                                                    
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='symmetric')         
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+size, j:j+size]
            result[i, j] = np.median(region)
    return result
def noise_analysis(image):                                         # Function to analyze noise by comparing the original image and its median-filtered version.
    median_filtered = manual_median_filter(image, size=5)
    noise = np.abs(image - median_filtered)                         # Calculating the noise as the absolute difference
    show_image("Noise Analysis", noise)
    return noise
def manual_arctan2(y, x):                                           # Function to manually compute the arctangent of y/x for gradient directions
    height, width = y.shape
    result = np.zeros_like(y, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            result[i, j] = np.arctan2(y[i, j], x[i, j]) * (180 / np.pi)  # Compute the angle in degrees
    return result
def gradient_direction(image):                                    #  Function to compute gradient directions of an image
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])       # Compute gradients in X & Y direction using Sobel Kernels
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = manual_convolution(image, sobel_x)
    gradient_y = manual_convolution(image, sobel_y)
    gradient_direction = manual_arctan2(gradient_y, gradient_x)        # Compute gradient directions.
    gradient_direction = np.mod(gradient_direction, 360)  
    return gradient_direction
def gradient_direction_comparison(original, suspected):                    # Function to compare gradient directions between the two images
    directions_original = gradient_direction(original)
    directions_suspected = gradient_direction(suspected)
    direction_diff = np.abs(directions_original - directions_suspected)         # Compute direction differences
    show_image("Original Image Gradient Directions", directions_original, cmap="jet")
    show_image("Suspected Image Gradient Directions", directions_suspected, cmap="jet")
    show_image("Gradient Direction Difference", direction_diff, cmap="jet")
    return direction_diff
def edge_comparison(original, suspected):                                  # Function to compare edges detected in two images
    edges_original = edge_detection(original)
    edges_suspected = edge_detection(suspected)
    edge_diff = np.abs(edges_original - edges_suspected)                   # Computing edge differences and clipping it in a valid range
    edge_diff = np.clip(edge_diff, 0, 255)
    show_image("Original Image Edges", edges_original)
    show_image("Suspected Image Edges", edges_suspected)
    show_image("Edge Difference", edge_diff)
    return edge_diff
def analyze_inpainting_with_original(original_path, suspected_path):        # Function to finally analyze inpainting by comparing the original and suspected images
    original = load_image(original_path, to_grayscale=True)                #Converting both the images to grayscale
    suspected = load_image(suspected_path, to_grayscale=True)
    pixel_difference_analysis(original, suspected)                        # Analyze pixel-level differences in both the images
    edge_comparison(original, suspected)                                  # Compare edges between images
    noise_analysis(original)                                              # Analyzing noise in the original image
    noise_analysis(suspected)                                             # Analyzing noise in the suspected image
    gradient_direction_comparison(original, suspected)                    # Comparing gradient directions
    print("Analysis complete. Results displayed.")
original_path = "Image2.jpg"                                              # Path to original image
suspected_path = "Image2inpainted.jpg"                                    # Path to suspected image
analyze_inpainting_with_original(original_path, suspected_path)
