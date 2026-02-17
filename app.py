import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Seitenconfig
st.set_page_config(page_title="Bildklassifikation", layout="centered")
st.title("Teachable Machine Bildklassifikation")

# Scientific notation deaktivieren
np.set_printoptions(suppress=True)

# Modell laden (einmal cachen für Performance)
@st.cache_resource
def load_ai_model():
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_ai_model()

# Datei-Upload
uploaded_file = st.file_uploader(
    "Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten wie im Teachable Machine Code
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    # Ergebnis anzeigen
    st.subheader("Ergebnis")
    st.write(f"**Klasse:** {class_name}")
    st.write(f"**Sicherheit:** {confidence_score:.2%}")
