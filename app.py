import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from github import Github
import base64
import json
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Bildklassifikation", layout="centered")
st.title("Teachable Machine Bildklassifikation")

accent_color = st.color_picker("Wähle eine Farbe", "#00bfff")

np.set_printoptions(suppress=True)

@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

@st.cache_resource
def connect_github():
    g = Github(st.secrets["GITHUB_TOKEN"])
    repo = g.get_repo(st.secrets["REPO_NAME"])
    return repo

model, class_names = load_ai_model()
repo = connect_github()

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    # Bild in Bytes umwandeln
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"images/{timestamp}.png"
    meta_path = f"metadata/{timestamp}.json"

    # Bild ins Repo committen
    repo.create_file(
        image_path,
        f"Upload image {timestamp}",
        base64.b64decode(encoded_image),
        branch="main"
    )

    metadata = {
        "filename": f"{timestamp}.png",
        "color": accent_color,
        "class": class_name,
        "confidence": confidence_score
    }

    repo.create_file(
        meta_path,
        f"Add metadata {timestamp}",
        json.dumps(metadata, indent=2),
        branch="main"
    )

    st.markdown(
        f"""
        <h2 style="color:{accent_color};">Ergebnis</h2>
        <p><b>Klasse:</b> {class_name}</p>
        <p><b>Sicherheit:</b> {confidence_score:.2%}</p>
        <p><b>Ins GitHub-Repo hochgeladen.</b></p>
        """,
        unsafe_allow_html=True
    )
