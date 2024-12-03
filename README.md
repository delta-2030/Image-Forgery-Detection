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

**Method to use the code:**
1. **Clone the Splicing Repository**
2. **Add Your Image**-Place the image you want to analyze in the repository folder. Change the image path to 
3. **Run the Script**
4. **Output Explanation**
**ELA Result**: Shows the error level analysis of the input image.
**Binary Mask**: Highlights regions with potential forgery.
**Mean Difference**: A value indicating the extent of compression inconsistencies.
Above 5: Potential forgery detected.
Below or equal to 5: No significant forgery detected.

**INPAINTING DETECTION**

Inpainting Detection is a forensic technique used to identify regions in an image that have been digitally altered or manipulated, specifically through inpainting. Inpainting involves reconstructing or filling parts of an image, often to remove unwanted objects or imperfections. This process can leave detectable artifacts, which the provided method highlights using pixel difference analysis, edge detection, noise analysis, and gradient direction comparison.

**Libraries used:**
**1. NumPy (numpy as np)**
**2. Pillow (PIL)**
**3. Matplotlib (matplotlib.pyplot as plt)**

**Method to use the code:**
1. **Clone the Inpainting Detection Repository**
2. **Add Your Target Images**-Place both the images (original and suspected) you want to analyze in the repository folder. Change the image path accordingly.  
3. **Run the Script**
4. **Output Explanation**
   The script will display:
Pixel difference map, edge maps of both images and their differences, noise maps of both images and gradient direction maps of both images and their differences. These outputs will provide insights into areas that were potentially modified in the suspected image. Pixel-difference map will reveal the potential pixel modifications in the image, edge map difference will show the added smoothness/layer over the image, noise analysis will potentially reveal any irregular noise patters in the image and the gradient different will point out the suspected regions in a heatmap.
