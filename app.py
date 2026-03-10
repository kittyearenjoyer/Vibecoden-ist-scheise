import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from datetime import datetime
from supabase import create_client, Client

st.set_page_config(page_title="AI Image Lab", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>

html, body {
background-color: #f7f9fc;
}

.block-container {
padding-top: 2rem;
max-width: 1200px;
}

h1 {
font-size: 48px;
font-weight: 700;
}

.subtitle {
color: #6b7280;
margin-bottom: 30px;
}

.glass {
background: white;
padding: 25px;
border-radius: 20px;
box-shadow: 0 10px 30px rgba(0,0,0,0.08);
margin-bottom: 25px;
}

.upload-area {
border: 2px dashed #d1d5db;
padding: 40px;
border-radius: 20px;
text-align: center;
background: #fafafa;
}

.gallery-card {
background: white;
padding: 10px;
border-radius: 16px;
box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}

.gallery-card img {
border-radius: 12px;
}

.progress {
height: 10px;
border-radius: 10px;
background: #e5e7eb;
}

.progress-fill {
height: 10px;
border-radius: 10px;
background: #6366f1;
}

.tag {
background: #eef2ff;
padding: 4px 10px;
border-radius: 20px;
font-size: 12px;
display: inline-block;
margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# Fundbüro")
st.markdown("<div class='subtitle'>Upload images • classify with AI • store automatically</div>", unsafe_allow_html=True)

# ---------------- AI MODEL ----------------
np.set_printoptions(suppress=True)

@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = [c.strip() for c in open("labels.txt").readlines()]
    return model, class_names

model, class_names = load_ai_model()

# ---------------- SUPABASE ----------------
@st.cache_resource
def connect_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key), url

supabase, supabase_url = connect_supabase()

# ---------------- COLOR ----------------
color_map = {
    "Rot": "#ef4444",
    "Grün": "#22c55e",
    "Orange": "#f97316"
}

selected_color_name = st.selectbox("Farbe auswählen", list(color_map.keys()))
selected_color_hex = color_map[selected_color_name]

# ---------------- UPLOAD ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

st.markdown("### Bild hochladen")

st.markdown("<div class='upload-area'>Ziehe ein Bild hier hinein oder wähle eine Datei</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CLASSIFICATION ----------------
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

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.markdown(f"### Ergebnis: {class_name}")

    st.markdown(f"""
    <div class="progress">
    <div class="progress-fill" style="width:{confidence*100}%"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"Confidence: **{confidence:.2%}**")

    st.markdown(f"<a href='{public_url}' target='_blank'>Bild öffnen</a>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- GALLERY ----------------
st.markdown("## Galerie")

response = supabase.table("image_meta").select("*").execute()
meta = response.data if response.data else []

if meta:

    classes = sorted(set(e["class"] for e in meta))
    colors = sorted(set(e["color"] for e in meta))

    col1, col2 = st.columns(2)

    with col1:
        filter_class = st.selectbox("Klasse", ["Alle"] + classes)

    with col2:
        filter_color = st.selectbox("Farbe", ["Alle"] + colors)

    filtered = [
        e for e in meta
        if (filter_class == "Alle" or e["class"] == filter_class)
        and (filter_color == "Alle" or e["color"] == filter_color)
    ]

    cols = st.columns(5)

    for i, entry in enumerate(filtered):

        with cols[i % 5]:

            st.markdown("<div class='gallery-card'>", unsafe_allow_html=True)

            st.image(entry["url"], use_container_width=True)

            st.markdown(
                f"<div class='tag'>{entry['class']}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div class='tag'>{entry['color']}</div>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Noch keine Bilder gespeichert.")
