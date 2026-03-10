import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from datetime import datetime
from supabase import create_client, Client

st.set_page_config(page_title="KI Bilddatenbank", layout="wide")

# ---------------- DESIGN ----------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
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

.upload-box {
    border: 2px dashed #555;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
}

.card {
    background: #1c1f26;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    margin-bottom: 15px;
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

st.title("🧠 KI Bildklassifikation + Supabase Storage")

st.markdown(
"""
Upload ein Bild → KI erkennt das Objekt → Bild wird automatisch gespeichert.

- 🤖 Klassifikation mit Keras  
- ☁️ Speicherung in Supabase  
- 🖼️ Galerie mit Filter
"""
)

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

# ---------------- UPLOAD ----------------
st.markdown("## 📤 Bild hochladen")
st.markdown('<div class="upload-box">Ziehe ein Bild hier hinein oder wähle eine Datei</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ---------------- KLASSIFIKATION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    arr = np.asarray(image_resized)
    normalized = (arr.astype(np.float32) / 127.5) - 1

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence = float(prediction[0][index])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"

    path = f"{class_name}/{selected_color_name}/{filename}"

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    file_bytes = buffer.getvalue()

    supabase.storage.from_("images").upload(
        path,
        file_bytes,
        {"content-type": "image/png"}
    )

    public_url = f"{supabase_url}/storage/v1/object/public/images/{path}"

    supabase.table("image_meta").insert({
        "filename": filename,
        "class": class_name,
        "color": selected_color_name,
        "upload_time": datetime.now().isoformat(),
        "url": public_url
    }).execute()

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

# ---------------- GALERIE ----------------
st.header("🖼️ Gespeicherte Bilder durchsuchen")

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
