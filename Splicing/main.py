import cv2
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

def ela(forged_image, error_image, new_quality=90):
    forged = Image.open(forged_image).convert('RGB')
    compressed_image = "temp_compressed.jpg"
    forged.save(compressed_image, 'JPEG', quality=new_quality)
    comp_image = Image.open(compressed_image)
    diff = ImageChops.difference(forged, comp_image)
    width, height = diff.size

    max_diff = 0
    for x in range(width):
        for y in range(height):
            r, g, b = diff.getpixel((x, y))
            pixel_diff = max(r, g, b) 
            if pixel_diff > max_diff:
                max_diff = pixel_diff
    if max_diff==0:
        scale=1

    else:

        scale = 255.0 / max_diff if max_diff > 0 else 1
    enhanced_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for x in range(width):
        for y in range(height):
            r, g, b = diff.getpixel((x, y))
            enhanced_r = int(min(r * scale, 255))
            enhanced_g = int(min(g * scale, 255))
            enhanced_b = int(min(b * scale, 255))
            enhanced_image[y, x] = (enhanced_r, enhanced_g, enhanced_b)

    ela_image = Image.fromarray(enhanced_image)
    ela_image.save(error_image)
    return np.array(ela_image)

def detect_forgery(forged_image):
    ela_image = ela(forged_image, "ela_output.jpg")
    gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_ela, 30, 255, cv2.THRESH_BINARY) 
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ela_image)
    plt.axis('off')
    plt.title("ELA Result")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.title("Binary Mask of Spliced Regions")

    plt.show()

    mean_diff = np.mean(gray_ela)
    print(f"Mean Difference: {mean_diff:.2f}")
    if mean_diff > 5:
        print("Potential forgery detected!")
    else:
        print("No forgery detected.")

image_path = "image3.jpg"
detect_forgery(image_path)
