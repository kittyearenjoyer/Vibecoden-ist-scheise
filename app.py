import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from datetime import datetime
from supabase import create_client, Client

st.set_page_config(page_title="KI Bilddatenbank", layout="wide")
st.title("KI Bildklassifikation + Supabase Storage")

# ---------------- AI MODEL ----------------
np.set_printoptions(suppress=True)

@st.cache_resource
def load_ai_model():
    model = load_model("keras_Model.h5", compile=False)
    class_names = [c.strip() for c in open("labels.txt").readlines()]
    return model, class_names

model, class_names = load_ai_model()

# ---------------- SUPABASE CLIENT ----------------
@st.cache_resource
def connect_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = connect_supabase()

# ---------------- FARBAUSWAHL ----------------
color_map = {
    "Rot": "#ff0000",
    "Grün": "#00aa00",
    "Orange": "#ff8800"
}
selected_color_name = st.selectbox("Farbe wählen", list(color_map.keys()))
selected_color_hex = color_map[selected_color_name]

# ---------------- UPLOAD + KLASSIFIKATION ----------------
st.header("Bild hochladen")
uploaded_file = st.file_uploader("Bild auswählen", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Modellvorbereitung
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    arr = np.asarray(image_resized)
    normalized = (arr.astype(np.float32)/127.5) - 1
    data = np.ndarray((1,224,224,3), dtype=np.float32)
    data[0] = normalized

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index])

    # Bild speichern in Supabase Storage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    path = f"{class_name}/{selected_color_name}/{filename}"

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    supabase.storage.from_("images").upload(path, buffer, {"content-type":"image/png"})

    # Public URL
    url = supabase.storage.from_("images").get_public_url(path).public_url

    # Metadaten in DB speichern
    supabase.table("image_meta").insert({
        "filename": filename,
        "class": class_name,
        "color": selected_color_name,
        "upload_time": datetime.now(),
        "url": url
    }).execute()

    st.markdown(f"""
    <h2 style="color:{selected_color_hex};">Klasse: {class_name}</h2>
    <p>Sicherheit: {confidence:.2%}</p>
    <p>Gespeichert unter: <a href='{url}' target='_blank'>{url}</a></p>
    """, unsafe_allow_html=True)

# ---------------- BILDER ANZEIGEN ----------------
st.header("Gespeicherte Bilder durchsuchen")

# Filter
meta = supabase.table("image_meta").select("*").execute().data

if meta:
    classes = sorted(set(e["class"] for e in meta))
    colors = sorted(set(e["color"] for e in meta))

    filter_class = st.selectbox("Nach Klasse filtern", ["Alle"] + classes)
    filter_color = st.selectbox("Nach Farbe filtern", ["Alle"] + colors)

    filtered = [
        e for e in meta
        if (filter_class == "Alle" or e["class"] == filter_class)
        and (filter_color == "Alle" or e["color"] == filter_color)
    ]

    cols = st.columns(4)
    for i, entry in enumerate(filtered):
        with cols[i%4]:
            st.image(entry["url"], caption=f"{entry['class']} | {entry['color']}")
else:
    st.info("Noch keine Bilder gespeichert.")
