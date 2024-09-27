import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

# Load the image
image_path = 'aether.jpg'  # Replace with your image file path
image = Image.open(image_path)

start_time = time.time()

image = image.convert("RGB")  # Convert to RGB if not already
image_np = np.array(image)

# Reshape the image to be a list of pixels
pixels = image_np.reshape(-1, 3)

# Set the number of colors (clusters) for quantization
num_colors = 8 # You can adjust this number

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(pixels)

# Replace each pixel by its cluster center
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_pixels = np.clip(compressed_pixels.astype('uint8'), 0, 255)

# Reshape back to the original image size
compressed_image_np = compressed_pixels.reshape(image_np.shape)

# Convert back to an image
compressed_image = Image.fromarray(compressed_image_np)
end_time = time.time()
execution_time = end_time - start_time

# Save the original and compressed images to calculate file sizes
original_image_path = 'original_image.jpg'
compressed_image_path = 'compressed_image.jpg'

# Save both images for comparison
image.save(original_image_path)
compressed_image.save(compressed_image_path)

# Get file sizes
original_size = os.path.getsize('aether.jpg')
compressed_size = os.path.getsize('compressed_image.jpg')

# Calculate compression ratio and percentage
compression_ratio = original_size / compressed_size
compression_percentage = 100 * (1 - (compressed_size / original_size))

# Display the original and compressed images side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f"Original Image\nSize: {original_size / 1024:.2f} KB")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image with {num_colors} colors\nSize: {compressed_size / 1024:.2f} KB")
plt.imshow(compressed_image)

plt.show()

# Display compression stats
print(f"Original Image Size: {original_size / 1024:.2f} KB")
print(f"Compressed Image Size: {compressed_size / 1024:.2f} KB")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Compression Percentage: {compression_percentage:.2f}%")
print(f"Execution Time:", execution_time)
