# Image-Forgery-Detection
**SPLICING**

Splicing forgery, a common type of digital image manipulation, involves copying and pasting regions from one or multiple images into another. Detecting such forgeries is crucial in digital forensics. 
The given presents a method based on Error Level Analysis (ELA) that utilizes image compression artifacts to identify tampered regions. By recompressing the image and analyzing 
differences in compression artifacts, this method effectively highlights inconsistencies that indicate possible forgery.

**Libraries used:**
**1. OpenCV (cv2)**
**Purpose:** OpenCV is an open-source computer vision library used for image processing.
**2. NumPy (numpy as np)**
**Purpose:** NumPy is used for handling arrays and numerical operations efficiently.
**3. Pillow (PIL)**
**Modules Used:** Image, ImageChops, ImageEnhance
**Purpose:** Pillow (PIL) is used for image manipulation and comparison.
4. Matplotlib (matplotlib.pyplot as plt)
**Purpose:** Matplotlib is a plotting library used for visualization.

**Concept**
Error Level Analysis (ELA) is a digital forensic technique used to detect image tampering, such as splicing or modification. It works by exploiting the fact that JPEG compression 
introduces varying levels of artifacts depending on how much an image region has been altered or recompressed.

In the Splicing folder, main.py contains the major code and the folder also includestet images which can be changed before running the code. 

**INPAINTING DETECTION**

Inpainting Detection is a forensic technique used to identify regions in an image that have been digitally altered or manipulated, specifically through inpainting. Inpainting involves reconstructing or filling parts of an image, often to remove unwanted objects or imperfections. This process can leave detectable artifacts, which the provided method highlights using pixel difference analysis, edge detection, noise analysis, and gradient direction comparison.

**Libraries used:**
**1. NumPy (numpy as np)**
**Purpose:** NumPy is used for handling arrays and numerical operations efficiently.
**2. Pillow (PIL)**
**Purpose:** Pillow (PIL) is used for image manipulation and comparison.
**3. Matplotlib (matplotlib.pyplot as plt)**
**Purpose:** Matplotlib is a plotting library used for visualization.
**4. SciPy (scipy.signal.convolve2d)**
**Purpose:** Used for applying convolution operations during image filtering and edge detection.

This method provides a robust pipeline for detecting tampered regions by leveraging pixel intensity differences, edge mismatches, noise inconsistencies, and gradient irregularities. Each step targets specific artifacts introduced during inpainting, making the approach comprehensive and effective.

In the Inpaiting Detection folder, main.py contains the major code and the folder also includes test images (both original and inpainted) which can be changed before running the code. 
