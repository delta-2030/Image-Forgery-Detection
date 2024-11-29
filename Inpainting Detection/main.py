import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_image(title, image, cmap="gray"):
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
def pixel_difference_analysis(original, suspected):
    diff = np.zeros_like(original, dtype=np.uint8)
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            diff_value = abs(int(original[i, j]) - int(suspected[i, j]))  
            diff[i, j] = np.clip(diff_value, 0, 255) 
    show_image("Pixel-Level Difference", diff)
    return diff
def manual_convolution(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(region * kernel) 
    return result
def edge_detection(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = manual_convolution(image, sobel_x)
    gradient_y = manual_convolution(image, sobel_y)
    edges = np.sqrt(gradient_x**2 + gradient_y**2)
    edges = (edges / edges.max()) * 255  
    edges = edges.astype(np.uint8)
    return edges
def manual_median_filter(image, size=5):
    height, width = image.shape
    pad = size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='symmetric')
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+size, j:j+size]
            result[i, j] = np.median(region)
    return result
def noise_analysis(image):
    median_filtered = manual_median_filter(image, size=5)
    noise = np.abs(image - median_filtered)
    show_image("Noise Analysis", noise)
    return noise
def manual_arctan2(y, x):
    height, width = y.shape
    result = np.zeros_like(y, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            result[i, j] = np.arctan2(y[i, j], x[i, j]) * (180 / np.pi)  
    return result
def gradient_direction(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = manual_convolution(image, sobel_x)
    gradient_y = manual_convolution(image, sobel_y)
    gradient_direction = manual_arctan2(gradient_y, gradient_x)
    gradient_direction = np.mod(gradient_direction, 360)  
    return gradient_direction
def gradient_direction_comparison(original, suspected):
    directions_original = gradient_direction(original)
    directions_suspected = gradient_direction(suspected)
    direction_diff = np.abs(directions_original - directions_suspected)
    show_image("Original Image Gradient Directions", directions_original, cmap="jet")
    show_image("Suspected Image Gradient Directions", directions_suspected, cmap="jet")
    show_image("Gradient Direction Difference", direction_diff, cmap="jet")
    return direction_diff
def edge_comparison(original, suspected):
    edges_original = edge_detection(original)
    edges_suspected = edge_detection(suspected)
    edge_diff = np.abs(edges_original - edges_suspected)
    edge_diff = np.clip(edge_diff, 0, 255)
    show_image("Original Image Edges", edges_original)
    show_image("Suspected Image Edges", edges_suspected)
    show_image("Edge Difference", edge_diff)
    return edge_diff
def analyze_inpainting_with_original(original_path, suspected_path):
    original = load_image(original_path, to_grayscale=True)
    suspected = load_image(suspected_path, to_grayscale=True)
    pixel_difference_analysis(original, suspected)
    edge_comparison(original, suspected)
    noise_analysis(original)
    noise_analysis(suspected)
    gradient_direction_comparison(original, suspected)
    print("Analysis complete. Results displayed.")
original_path = "C:\\Users\\Aryan\\Downloads\\Dhoni800x800 (1).jpg" 
suspected_path = "C:\\Users\\Aryan\\Downloads\\Dhoni800x800.jpg"  
analyze_inpainting_with_original(original_path, suspected_path)
