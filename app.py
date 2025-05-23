import streamlit as st
import pickle
import numpy as np
import cv2
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title(" Digit Classifier")
st.write("Draw a digit ")

# Load trained model and scaler
model = pickle.load(open("mnist_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Improved image preprocessing function
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    padded = padded / 255.0

    features = hog(padded, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return scaler.transform([features])

# --- Drawing canvas ---
st.subheader("Draw a Digit Below")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Process Canvas Drawing ---
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    features = preprocess_image(img_bgr)
    prediction = model.predict(features)[0]

    st.image(img_bgr, caption="Drawn Digit", use_container_width=True)
    st.write(f"**Predicted Digit from Drawing:** `{prediction}`")

# --- Image Uploader ---
st.subheader("Or Upload a Digit Image")
uploaded_file = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    features = preprocess_image(image)
    prediction = model.predict(features)[0]

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Predicted Digit from Upload:** `{prediction}`")
