# --- Module Imports ---
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import random

# --- Step 1: Load and Prepare Data ---
print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1)
images, labels = mnist.data.values, mnist.target.astype(np.uint8)

# Reshape each flat image to 28x28 pixels
images_reshaped = images.reshape(-1, 28, 28)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_reshaped, labels, test_size=0.2, random_state=0)

# --- Step 2: HOG Feature Extraction ---
def compute_hog_features(img_array):
    features = [hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for img in img_array]
    return np.array(features)

print("Extracting HOG features...")
X_train_hog = compute_hog_features(X_train)
X_test_hog = compute_hog_features(X_test)

# Normalize features
scaler = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

# --- Step 3: Model Training and Evaluation ---
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    return model, accuracy

print("\nTraining models...")
log_model, acc_log = evaluate_model(LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial"),
                                    X_train_hog, y_train, X_test_hog, y_test, "Logistic Regression") # Changed X_train_scaled to X_train_hog and X_test_scaled to X_test_hog

nb_model, acc_nb = evaluate_model(GaussianNB(),
                                  X_train_hog, y_train, X_test_hog, y_test, "Naive Bayes") # Changed X_train_scaled to X_train_hog and X_test_scaled to X_test_hog

dt_model, acc_dt = evaluate_model(DecisionTreeClassifier(max_depth=20, random_state=0),
                                  X_train_hog, y_train, X_test_hog, y_test, "Decision Tree") # Changed X_train_scaled to X_train_hog and X_test_scaled to X_test_hog

# --- Step 4: Test Prediction on Random Image ---
idx = random.randint(0, len(X_test) - 1)
sample_image = X_test[idx]
sample_label = y_test.iloc[idx]
sample_hog = hog(sample_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False).reshape(1, -1)
sample_scaled = scaler.transform(sample_hog)

predicted = log_model.predict(sample_scaled)[0]

plt.imshow(sample_image, cmap='gray')
plt.title(f"Predicted: {predicted}, Actual: {sample_label}")
plt.axis('off')
plt.show()

# --- Step 5: Save Model ---
with open("mnist_model.pkl", "wb") as f:
    pickle.dump(log_model, f)
print("Trained model saved as mnist_model.pkl")

import pickle

# Save the scaler used to normalize HOG features
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully!")

!pip install streamlit
!pip install opencv-python
!pip install scikit-image
!pip install scikit-learn
import streamlit as st
import pickle
import numpy as np
import cv2
from skimage.feature import hog

st.title("MNIST Digit Classifier 🎨")
st.write("Upload a handwritten digit image to classify it.")

# Load the trained model and scaler
model = pickle.load(open("mnist_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Improved preprocessing
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28 without interpolation artifacts
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert colors if background is white
    if np.mean(resized) > 127:
        resized = 255 - resized

    # Normalize pixel values
    normalized = resized / 255.0

    # HOG feature extraction
    features = hog(normalized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    scaled_features = scaler.transform([features])
    return scaled_features

# Upload section
uploaded_file = st.file_uploader("Upload a digit image (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process and predict
    try:
        features = preprocess_image(image)
        prediction = model.predict(features)[0]
        st.success(f"**Predicted Digit:** {prediction}")
    except Exception as e:
        st.error(f"Error in processing image: {str(e)}")

app_code = """
import streamlit as st
import pickle
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

st.title("MNIST Digit Classifier 🎨")
st.write("Upload a handwritten digit image (JPG/PNG). It will be processed and classified using a HOG + ML model.")

# Load trained model and scaler
model = pickle.load(open("mnist_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Improved image preprocessing function
def preprocess_image(uploaded_image):
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Invert if background is white
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours and crop to bounding box
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize to 20x20 and pad to 28x28
    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    padded = padded / 255.0  # normalize

    # Extract HOG features
    features = hog(padded, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return scaler.transform([features])  # apply saved scaler

# File uploader
uploaded_file = st.file_uploader("Upload a Digit Image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and decode the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess and predict
    features = preprocess_image(image)
    prediction = model.predict(features)[0]

    # Display results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Predicted Digit:** `{prediction}`")

"""

# Write to app.py
with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py saved!")
