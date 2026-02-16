import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

st.set_page_config(page_title="TensorFlow Bildanalyse", page_icon="🧠", layout="centered")
st.title("🧠 Bild-Upload + KI-Auswertung (TensorFlow)")
st.write(
    "Lade ein Bild hoch. Das vortrainierte MobileNetV2-Modell klassifiziert das Motiv "
    "und zeigt die Top-3 Vorhersagen an."
)


@st.cache_resource
def load_model() -> tf.keras.Model:
    return MobileNetV2(weights="imagenet")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
model = load_model()

uploaded_file = st.file_uploader(
    "Bild auswählen", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    resized = image.resize((224, 224))
    array = np.array(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = preprocess_input(batch)

    with st.spinner("Analysiere Bild mit TensorFlow..."):
        preds = model.predict(batch, verbose=0)
        top3 = decode_predictions(preds, top=3)[0]

    st.subheader("Top-3 KI-Vorhersagen")
    for rank, (_, label, confidence) in enumerate(top3, start=1):
        st.write(f"**{rank}. {label}** – {confidence * 100:.2f}%")
else:
    st.info("Bitte ein Bild hochladen, um die Analyse zu starten.")
