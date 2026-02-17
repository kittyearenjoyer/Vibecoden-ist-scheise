import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from github import Github
from io import BytesIO
from datetime import datetime
import base64

st.set_page_config(page_title="KI Bilddatenbank", layout="wide")
st.title("KI Bildklassifikation + Repository Datenbank")

# ---------- AI MODEL ----------

np.set_printoptions(suppress=True)

@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = [c.strip() for c in open("labels.txt").readlines()]
    return model, class_names

model, class_names = load_ai_model()

# ---------- GITHUB ----------

@st.cache_resource
def connect_github():
    g = Github(st.secrets["GITHUB_TOKEN"])
    repo = g.get_repo(st.secrets["REPO_NAME"])
    return repo

repo = connect_github()

# ---------- FARBAUSWAHL ----------

color_map = {
    "Rot": "#ff0000",
    "Grün": "#00aa00",
    "Orange": "#ff8800"
}

selected_color_name = st.selectbox(
    "Farbe wählen",
    list(color_map.keys())
)

selected_color_hex = color_map[selected_color_name]

# ---------- UPLOAD + KLASSIFIKATION ----------

st.header("Bild hochladen")

uploaded_file = st.file_uploader(
    "Bild auswählen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Vorbereitung fürs Modell
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence = float(prediction[0][index])

    # ---------- SPEICHERN INS REPO ----------

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"database/{class_name}/{selected_color_name}/{timestamp}.png"

    repo.create_file(
        path,
        f"Upload {class_name} {selected_color_name} {timestamp}",
        img_bytes,
        branch="main"
    )

    st.markdown(
        f"""
        <h2 style="color:{selected_color_hex};">
        Klasse: {class_name}
        </h2>
        <p>Sicherheit: {confidence:.2%}</p>
        <p>Gespeichert unter:<br>{path}</p>
        """,
        unsafe_allow_html=True
    )

# ---------- DATENBANK ANZEIGE ----------

st.header("Gespeicherte Bilder durchsuchen")

def list_repo_files(folder):
    try:
        contents = repo.get_contents(folder)
        files = []
        for item in contents:
            if item.type == "dir":
                files.extend(list_repo_files(item.path))
            else:
                files.append(item)
        return files
    except:
        return []

files = list_repo_files("database")

# Attribute extrahieren
entries = []

for f in files:
    parts = f.path.split("/")
    if len(parts) >= 4:
        entries.append({
            "file": f,
            "class": parts[1],
            "color": parts[2]
        })

if entries:

    classes = sorted(set(e["class"] for e in entries))
    colors = sorted(set(e["color"] for e in entries))

    filter_class = st.selectbox(
        "Nach Klasse filtern",
        ["Alle"] + classes
    )

    filter_color = st.selectbox(
        "Nach Farbe filtern",
        ["Alle"] + colors
    )

    filtered = [
        e for e in entries
        if (filter_class == "Alle" or e["class"] == filter_class)
        and (filter_color == "Alle" or e["color"] == filter_color)
    ]

    cols = st.columns(4)

    for i, entry in enumerate(filtered):
        img_data = base64.b64decode(entry["file"].content)
        img = Image.open(BytesIO(img_data))

        with cols[i % 4]:
            st.image(img, caption=f"{entry['class']} | {entry['color']}")

else:
    st.info("Noch keine Bilder gespeichert.")
