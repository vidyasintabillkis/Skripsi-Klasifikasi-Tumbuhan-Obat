import streamlit as st
import numpy as np
import tensorflow as tf
import time, requests, tempfile
from PIL import Image, UnidentifiedImageError
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Plantifier",
    page_icon="🌱",
)

# Fungsi untuk download model dari Google Drive
@st.cache_resource(show_spinner=False)
def load_model_from_github(url, model_name):
    try:
        model_path = tempfile.NamedTemporaryFile(delete=False, suffix='.h5').name
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        model = tf.keras.models.load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Gagal memuat model {model_name}: {str(e)}")
        return None

EFFICIENTNET_MODEL_URL = "https://github.com/vidyasintabillkis/SKRIPSI/releases/download/v1.0.0/efficientnet_9.h5"
XCEPTION_MODEL_URL = "https://github.com/vidyasintabillkis/SKRIPSI/releases/download/v1.0.0/xception_9.h5"

model_efficientnet = load_model_from_github(EFFICIENTNET_MODEL_URL, "EfficientNetV2")
model_xception = load_model_from_github(XCEPTION_MODEL_URL, "Xception")

if model_efficientnet is None or model_xception is None:
    st.error("Aplikasi tidak dapat berjalan tanpa model. Silakan hubungi administrator.")
    st.stop()

#Label kelas
labels = [ 
    "Bidara", "Binahong", "Cincau Hijau", "Kejibeling", "Kelor",
    "Ketapang", "Pulai", "Salam", "Sambung Nyawa", "Sirih"
]

#Data taksonomi dan gambar pohon
plant_info = {
    "Bidara": {
        "scientific_name": "Ziziphus mauritiana",
        "nama_lain": ", Widara, Dara, Bukkol, Bekul, Bedara, Kalangga",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil", "Rosid"],
            "Ordo": "Rhamnales",
            "Famili": "Rhamnaceae",
            "Genus": "Ziziphus",
            "Spesies": "Z. mauritiana"
        },
        "image_path": "gambar/pohon/Bidara.jpeg"
    },
    "Binahong": {
        "scientific_name": "Anredera cordifolia",
        "nama_lain": ", Piahong, Piyahong",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta"],
            "Ordo": "Caryophyllales",
            "Famili": "Basellaceae",
            "Genus": "Anredera",
            "Spesies": "Anredera cordifolia"
        },
        "image_path": "gambar/pohon/Binahong.jpg"
    },
    "Cincau Hijau": {
        "scientific_name": "Cyclea barbata",
        "nama_lain": ", Camcao, Camcauh, Juju, Kepleng, Krotok, Tahulu",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil"],
            "Ordo": "Ranunculales",
            "Famili": "Menispermaceae",
            "Genus": "Cyclea",
            "Spesies": "C. barbata"
        },
        "image_path": "gambar/pohon/Cincau.jpeg"
    },
    "Kejibeling": {
        "scientific_name": "Strobilanthes crispus",
        "nama_lain": ", Kecibeling, Picah Beling, Ki Beling, Enyoh Kelo",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Angiosperma", "Eudikotil", "Asterid"],
            "Ordo": "Lamiales",
            "Famili": "Acanthaceae",
            "Genus": "Strobilanthes",
            "Spesies": "S. crispa"
        },
        "image_path": "gambar/pohon/Kejibeling.jpeg"
    },
    "Kelor": {
        "scientific_name": "Moringa oleifera",
        "nama_lain": ", Limaran, Merunggai, Moringa",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil", "Rosid"],
            "Ordo": "Brassicales",
            "Famili": "Moringaceae",
            "Genus": "Moringa",
            "Spesies": "M. oleifera"
        },
        "image_path": "gambar/pohon/Kelor.jpg"
    }, 
    "Ketapang": {
        "scientific_name": "Terminalia catappa",
        "nama_lain": ", Katapang, Hatapang, Talisei, Tiliso, Sarisa, Lisa, Kalis",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil", "Rosid"],
            "Ordo": "Myrtales",
            "Famili": "Combretaceae",
            "Genus": "Terminalia",
            "Spesies": "T. catappa"
        },
        "image_path": "gambar/pohon/Ketapang.jpg"
    },
    "Pulai": {
        "scientific_name": "Alstonia scholaris",
        "nama_lain": ", Pule, Kayu Gabus, Lame, Lamo, Jelutung",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil", "Asterid"],
            "Ordo": "Gentianales",
            "Famili": "Apocynaceae",
            "Genus": "Alstonia",
            "Spesies": "A. scholaris"
        },
        "image_path": "gambar/pohon/Pulai.jpeg"
    },
    "Salam": {
        "scientific_name": "Syzygium polyanthum",
        "nama_lain": ", Ubar Seribu, Serai Kayu, Salam Kojo",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil", "Rosid"],
            "Ordo": "Myrtales",
            "Famili": "Myrtaceae",
            "Genus": "Syzygium",
            "Spesies": "S. polyanthum"
        },
        "image_path": "gambar/pohon/Salam.jpg"
    },
    "Sambung Nyawa": {
        "scientific_name": "Gynura procumbens",
        "nama_lain": ", Daun Dewa, Akar Sebiak",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Eudikotil", "	Asterid"],
            "Ordo": "Asterales",
            "Famili": "Asteraceae",
            "Genus": "Gynura",
            "Spesies": "G. procumbens"
        },
        "image_path": "gambar/pohon/SambungNyawa.jpg"
    },
    "Sirih": {
        "scientific_name": "Piper betle L.",
        "nama_lain": ", Suruh, Lu'at, Sireh, Bido, Base, Amo",
        "taxonomy": {
            "Kingdom": "Plantae",
            "Klad": ["Tracheophyta", "Angiospermae", "Magnoliid"],
            "Ordo": "Piperales",
            "Famili": "Piperaceae",
            "Genus": "Piper",
            "Spesies": "P. betle"
        },
        "image_path": "gambar/pohon/Sirih.jpg"
    }
}

#Fungsi prediksi dengan threshold
def predict_with_threshold(model, img_array, threshold=0.65):
    start_time = time.time()
    prediction = model.predict(img_array)
    end_time = time.time()
    execution_time = end_time - start_time

    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    confidence_percent = confidence * 100

    if confidence < threshold:
        return "Kelas Tidak Dikenal", confidence_percent, execution_time
    else:
        return labels[predicted_class[0]], confidence_percent, execution_time

#Preprocessing gambar
def preprocess_image(image):
    img_size = (224, 224)  # Sesuaikan ukuran input model
    image = image.resize(img_size)
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1.0  # Normalisasi sesuai model
    image = np.expand_dims(image, axis=0)
    return image

def show_plant_info(label, conf):
    if label in plant_info:
        scientific_name = plant_info[label]["scientific_name"]
        nama_lain = plant_info[label]["nama_lain"]
        taxonomy = plant_info[label]["taxonomy"]
        image_path = plant_info[label]["image_path"]

        col1, spacer, col2 = st.columns([1, 0.1, 1])

        with col2: 
            st.subheader(f"Klasifikasi Ilmiah")
            st.write(f"**Nama** : {label}{nama_lain}") 
            st.write(f"**Nama Latin** : {scientific_name}")
            st.write(f"**Kingdom** : {taxonomy['Kingdom']}")
            st.write(f"**Klad** : {', '.join(taxonomy['Klad'])}")
            st.write(f"**Ordo**: {taxonomy['Ordo']}")
            st.write(f"**Famili** : {taxonomy['Famili']}")
            st.write(f"**Genus** : {taxonomy['Genus']}")
            st.write(f"**Spesies** : {taxonomy['Spesies']}")
            st.markdown(f"**Keakuratan** : {conf:.2f}%")

        with col1:
            try:
                tree_image = Image.open(image_path)
                st.image(tree_image, use_container_width=True)
            except FileNotFoundError:
                st.error(f"⚠️ Gambar pohon untuk {label} tidak ditemukan.")

#Tampilan navbar
selected = option_menu(
    menu_title=None,
    options=["Panduan", "Klasifikasi"],
    icons=["house", "search"],
    orientation="horizontal",
)

if selected == "Panduan":
    st.title("Halo, Selamat Datang!")

    st.markdown(
        """
        Untuk menggunakan website ini, silakan ikuti panduan berikut: 
        #### 🛠️ Cara Pengunaan
        - Buka tab klasifikasi
        - Unggah gambar daun tumbuhan obat yang ingin diklasifikasikan 
        - Klik tombol Klasifikasi
        - Lihat hasil klasifikasi dari dua model yang dipakai untuk mengenali gambar
        #### ⚠️ Catatan Penting
        Pastikan gambar yang diunggah merupakan **daun tunggal dengan latar belakang 
        berwarna putih** untuk memastikan hasil prediksi yang lebih akurat.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image("gambar/Sirih.jpg", caption="**Contoh gambar yang benar**")
    with col2:
        st.image("gambar/salah.jpg", caption="**Contoh gambar yang salah**")
    
elif selected == "Klasifikasi":
    st.title("Klasifikasi Tumbuhan Obat")
    uploaded_file = st.file_uploader("Silakan unggah gambar daun sesuai panduan (jpg, jpeg, png)", type=None)

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        allowed_exts = ["jpg", "jpeg", "png"]

        if file_ext not in allowed_exts:
            st.error("❌ Format file tidak didukung. Silakan unggah file dengan format JPG, JPEG, atau PNG.")
        else:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, width=300)

                if st.button("Prediksi", type="primary"):
                    img_array = preprocess_image(image)
                    label1, conf1, time1 = predict_with_threshold(model_efficientnet, img_array)
                    label2, conf2, time2 = predict_with_threshold(model_xception, img_array)
                    tab1, tab2 = st.tabs(["Klasifikasi Model EfficientNetV2B0", "Klasifikasi Model Xception"])

                    with tab1:
                        if label1 == "Kelas Tidak Dikenal":
                            st.error("⚠️ Mohon maaf, sistem tidak dapat mengenali tumbuhan ini (EfficientNet).")
                        else:
                            show_plant_info(label1, conf1)
                            # st.caption(f"Hasil klasifikasi menggunakan model EfficientNet")

                    with tab2:
                        if label2 == "Kelas Tidak Dikenal":
                            st.error("⚠️ Mohon maaf, sistem tidak dapat mengenali tumbuhan ini (Xception).")
                        else:
                            show_plant_info(label2, conf2)
                            # st.caption(f"Hasil klasifikasi menggunakan model Xception")

            except UnidentifiedImageError:
                st.error("❌ File yang diunggah bukan gambar yang valid atau gambar corrupt. Silakan unggah ulang.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan saat memproses gambar: {e}")
