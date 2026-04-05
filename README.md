# SSIM-PSNR-and-MSE

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (use your Desktop path)
image = cv2.imread('C:/Users/elena/Desktop/mushroom.jpg', cv2.IMREAD_GRAYSCALE)

import cv2
import matplotlib.pyplot as plt

# Load image in color
image = cv2.imread('C:/Users/elena/Desktop/mushroom.jpg')

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Original Image')
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original image in grayscale
original = cv2.imread('C:/Users/elena/Desktop/mushroom.jpg', cv2.IMREAD_GRAYSCALE)
if original is None:
    raise ValueError("Image not found. Check the path!")


# Add small Gaussian noise for MSE
mse_mod = cv2.add(original, np.random.normal(0, 10, original.shape).astype(np.uint8))
# Add stronger noise for PSNR illustration
psnr_mod = cv2.add(original, np.random.normal(0, 30, original.shape).astype(np.uint8))
# Blur image for SSIM illustration
ssim_mod = cv2.GaussianBlur(original, (5, 5), 0)

# Metric functions
def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(m))

def ssim(a, b):
    C1 = 6.5025
    C2 = 58.5225

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(a**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(b**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(a*b, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# Compute metrics
mse_val = mse(original, mse_mod)
psnr_val = psnr(original, psnr_mod)
ssim_val = ssim(original, ssim_mod)

# Print metrics
print(f"MSE: {mse_val:.2f}")
print(f"PSNR: {psnr_val:.2f}")
print(f"SSIM: {ssim_val:.4f}")

# Visualize images
plt.figure(figsize=(12, 6))

# Original
plt.subplot(2, 4, 1)
plt.imshow(original, cmap='gray')
plt.title("Original")
plt.axis('off')

# MSE comparison
plt.subplot(2, 4, 2)
plt.imshow(mse_mod, cmap='gray')
plt.title(f"MSE Image\n{mse_val:.2f}")
plt.axis('off')

# PSNR comparison
plt.subplot(2, 4, 3)
plt.imshow(psnr_mod, cmap='gray')
plt.title(f"PSNR Image\n{psnr_val:.2f}")
plt.axis('off')

# SSIM comparison
plt.subplot(2, 4, 4)
plt.imshow(ssim_mod, cmap='gray')
plt.title(f"SSIM Image\n{ssim_val:.4f}")
plt.axis('off')

plt.tight_layout()
plt.show()
