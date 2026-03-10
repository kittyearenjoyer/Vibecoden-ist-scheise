import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from datetime import datetime
from supabase import create_client, Client

st.set_page_config(page_title="KI Bilddatenbank", layout="wide")

st.markdown("""
<style>

.main {
    background-color: #000080;
}

h1, h2, h3 {
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg,#ff4b4b,#ff8800);
    color: white;
    border: none;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
}

.upload-box {
    border: 2px dashed #555;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.card {
    background: #1c1f26;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

.card img {
    border-radius: 10px;
}

.confbar {
    height: 8px;
    border-radius: 10px;
    background: #333;
}

.confbar-fill {
    height: 8px;
    border-radius: 10px;
    background: linear-gradient(90deg,#00ff9d,#00bfff);
}

</style>
""", unsafe_allow_html=True)

st.title("KI Bildklassifikation + Supabase Storage")

# ---------------- AI MODEL ----------------
np.set_printoptions(suppress=True)

@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = [c.strip() for c in open("labels.txt").readlines()]
    return model, class_names

model, class_names = load_ai_model()

# ---------------- SUPABASE CLIENT ----------------
@st.cache_resource
def connect_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key), url

supabase, supabase_url = connect_supabase()

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
uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild laden
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Modellvorbereitung
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    arr = np.asarray(image_resized)
    normalized = (arr.astype(np.float32) / 127.5) - 1
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index])

    # Dateiname & Pfad
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    path = f"{class_name}/{selected_color_name}/{filename}"

    # Bild in Bytes umwandeln (WICHTIG für Supabase!)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    file_bytes = buffer.getvalue()

    # Upload zu Supabase Storage (nur EINMAL!)
    supabase.storage.from_("images").upload(
        path,
        file_bytes,
        {"content-type": "image/png"}
    )

    # Public URL generieren
    public_url = f"{supabase_url}/storage/v1/object/public/images/{path}"

    # Metadaten in Datenbank speichern
    supabase.table("image_meta").insert({
        "filename": filename,
        "class": class_name,
        "color": selected_color_name,
        "upload_time": datetime.now().isoformat(),
        "url": public_url
    }).execute()

    # Ausgabe
    st.markdown(f"""
<div class="card">

<h2 style="color:{selected_color_hex};">
🧠 Klasse: {class_name}
</h2>

<p><b>Confidence:</b> {confidence:.2%}</p>

<div class="confbar">
<div class="confbar-fill" style="width:{confidence*100}%"></div>
</div>

<br>

<a href="{public_url}" target="_blank">🔗 Bild öffnen</a>

</div>
""", unsafe_allow_html=True)


# ---------------- BILDER ANZEIGEN ----------------
st.header("Gespeicherte Bilder durchsuchen")

response = supabase.table("image_meta").select("*").execute()
meta = response.data if response.data else []

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
    with cols[i % 4]:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.image(entry["url"], use_container_width=True)

        st.markdown(
            f"<center><b>{entry['class']}</b><br>{entry['color']}</center>",
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Noch keine Bilder gespeichert.")
