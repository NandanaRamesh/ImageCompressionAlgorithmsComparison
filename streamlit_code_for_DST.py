import streamlit as st
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import io

# Functions for RGB <-> YCbCr conversion and DCT processing
def rgb2ycbcr(image):
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.460525 * g - 0.081975 * b
    return y, cb, cr

def ycbcr2rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.clip(np.stack((r, g, b), axis=-1), 0, 255).astype(np.uint8)


def create_quantization_matrix(quality_factor):
    base_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])

    scale = 5000 / quality_factor if quality_factor < 50 else 200 - quality_factor * 2
    quantization_matrix = np.round((base_matrix * scale) / 100).astype(int)
    return quantization_matrix


def block_dct(image):
    h, w = image.shape
    dct_image = np.zeros_like(image)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i + 8, j:j + 8]
            dct_image[i:i + 8, j:j + 8] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    return dct_image


def block_idct(dct_image):
    h, w = dct_image.shape
    idct_image = np.zeros_like(dct_image)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_image[i:i + 8, j:j + 8]
            idct_image[i:i + 8, j:i + 8] = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    return idct_image


def compress_image(image_path, quality_factor):
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image.width // 8 * 8, image.height // 8 * 8))  # Ensure dimensions are divisible by 8
    img_array = np.array(image)

    # Convert RGB to YCbCr
    y, cb, cr = rgb2ycbcr(img_array)

    # Perform DCT in 8x8 blocks
    dct_y = block_dct(y)
    dct_cb = block_dct(cb)
    dct_cr = block_dct(cr)

    # Quantization
    quantization_matrix = create_quantization_matrix(quality_factor)
    q_dct_y = np.zeros_like(dct_y)
    q_dct_cb = np.zeros_like(dct_cb)
    q_dct_cr = np.zeros_like(dct_cr)

    # Apply quantization to each block
    for i in range(0, dct_y.shape[0], 8):
        for j in range(0, dct_y.shape[1], 8):
            q_dct_y[i:i + 8, j:j + 8] = np.round(dct_y[i:i + 8, j:j + 8] / quantization_matrix)
            q_dct_cb[i:i + 8, j:j + 8] = np.round(dct_cb[i:i + 8, j:j + 8] / quantization_matrix)
            q_dct_cr[i:i + 8, j:j + 8] = np.round(dct_cr[i:i + 8, j:j + 8] / quantization_matrix)

    # Inverse DCT in 8x8 blocks
    idct_y = np.zeros_like(q_dct_y)
    idct_cb = np.zeros_like(q_dct_cb)
    idct_cr = np.zeros_like(q_dct_cr)

    for i in range(0, q_dct_y.shape[0], 8):
        for j in range(0, q_dct_y.shape[1], 8):
            idct_y[i:i + 8, j:j + 8] = idct(idct(q_dct_y[i:i + 8, j:j + 8] * quantization_matrix, axis=0, norm='ortho'),
                                            axis=1, norm='ortho')
            idct_cb[i:i + 8, j:j + 8] = idct(
                idct(q_dct_cb[i:i + 8, j:j + 8] * quantization_matrix, axis=0, norm='ortho'), axis=1, norm='ortho')
            idct_cr[i:i + 8, j:j + 8] = idct(
                idct(q_dct_cr[i:i + 8, j:j + 8] * quantization_matrix, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Convert back to RGB
    compressed_image = ycbcr2rgb(idct_y, idct_cb, idct_cr)

    return Image.fromarray(compressed_image)


# Streamlit app
st.title("Image Compression using DCT")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# Quality factor input
quality_factor = st.slider("Select Quality Factor", min_value=5, max_value=90, value=5)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display original image
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the size of the original image
    original_size = uploaded_file.size  # Size in bytes

    # Compress image (passing uploaded_file directly)
    compressed_image = compress_image(uploaded_file, quality_factor)

    # Get the size of the compressed image
    buf = io.BytesIO()
    compressed_image.save(buf, format="JPEG")
    compressed_size = buf.tell()  # Size in bytes
    buf.seek(0)  # Reset buffer position to read again

    # Display compressed image
    st.subheader(f"Compressed Image (Quality Factor: {quality_factor})")
    st.image(compressed_image, caption="Compressed Image", use_column_width=True)

    # Display sizes of original and compressed images
    st.write(f"Original Image Size: {original_size / 1024:.2f} KB")
    st.write(f"Compressed Image Size: {compressed_size / 1024:.2f} KB")

    # Option to download compressed image
    byte_im = buf.getvalue()
    st.download_button("Download Compressed Image", byte_im, file_name="compressed_image.jpg", mime="image/jpeg")
