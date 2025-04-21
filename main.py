import streamlit as st
from ultralytics import YOLO
from PIL import Image
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import io

# ---- Title and Header ----
st.set_page_config(page_title="Golf Equipment Detection", layout="centered")
st.title("🏌️ Golf Equipment Detection App")
st.markdown("Upload an image of golf equipment, and the YOLO model will identify and highlight the items. The app will also extract text using Azure OCR and predict the price.")

# ---- Sidebar Configuration ----
st.sidebar.header("⚙️ Model Configuration")
confidence_value = float(st.sidebar.slider("Confidence Threshold", min_value=25, max_value=100, value=50)) / 100
model_path = "best.pt"

# Azure OCR Configuration
st.sidebar.header("🔍 Azure OCR Configuration")
subscription_key = st.sidebar.text_input("Azure Subscription Key", type="password")
endpoint = st.sidebar.text_input("Azure Endpoint", value="https://eastus.api.cognitive.microsoft.com/")

# ---- Load the Model ----
@st.cache_resource
def load_model(path):
    return YOLO(path)

@st.cache_resource
def load_p_model():
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    ohe = joblib.load("ohe_encoder.pkl")
    ord_enc = joblib.load("ordinal_encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return (model, ohe, ord_enc, feature_cols)

xgb_model, ohe, ord_enc, feature_cols = load_p_model()
model = load_model(model_path)

# ---- Mapping YOLO Classes to Price Prediction Types ----
yolo_to_price_type = {
    "driver": "driver",
    "fairway_wood": "bois",
    "golf_bag": None,
    "golf_ball": None,
    "hybrid": "hybride",
    "iron": "fers",
    "putter": "putter",
    "wedge": "wedge"
}

# ---- Define Constants at Top Level ----
brands = ["CALLAWAY", "COBRA", "PXG", "BENROSS", "TAYLORMADE", "TITLEIST", "TOULON DESIGN", "SRIXON", "PING", "CLEVELAND", "SCOTTY CAMERON", "MIURA", "MASTERS", "XXIO", "FEEL GOLF", "ODYSSEY", "HONMA", "NIKE", "MIZUNO", "WILSON", "BETTINARDI", "JUCAD", "BOBBY JONES", "INESIS", "HERRIA", "BRIDGESTONE", "WILSON STAFF", "PEGG", "LONGRIDGE", "POWERBILT", "US KIDS"]
etat_options = ["Moyen", "Bon", "Très bon", "Neuf", "Excellent"]

# ---- Azure OCR Setup ----
def initialize_ocr_client():
    if subscription_key and endpoint:
        credentials = CognitiveServicesCredentials(subscription_key)
        return ComputerVisionClient(endpoint, credentials)
    return None

# ---- Initialize Variables ----
marque_detected = None
etat_detected = None
annee_detected = None
detected_label = None
mapped_type = None

# ---- Image Upload Section ----
st.markdown("---")
st.subheader("📸 Upload Your Image")

uploaded_file = st.file_uploader("Supported formats: PNG, JPG, JPEG, WEBP, BMP", type=["png", "jpg", "jpeg", "webp", "bmp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    img = Image.open(uploaded_file).convert("RGB")
    col1.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(img, conf=confidence_value)
        result_img = results[0].plot()[:, :, ::-1]

    col2.image(result_img, caption="🧠 Detected Equipment", use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detection Summary")
    boxes = results[0].boxes
    if boxes is not None and boxes.shape[0] > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id] if model.names else f"Class {cls_id}"
            st.markdown(f"**{label}** - Confidence: `{conf:.2%}`")
            detected_label = label.lower()
    else:
        st.info("No objects detected at the selected confidence threshold.")

    mapped_type = yolo_to_price_type.get(detected_label) if detected_label else None
    if mapped_type is None and detected_label:
        st.warning(f"Detected item '{detected_label}' is not supported for price prediction.")

    ocr_client = initialize_ocr_client()
    if ocr_client:
        with st.spinner("Extracting text with Azure OCR..."):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            ocr_result = ocr_client.recognize_printed_text_in_stream(img_byte_arr)
            extracted_text = []
            for region in ocr_result.regions:
                for line in region.lines:
                    for word in line.words:
                        extracted_text.append(word.text.upper())

            st.markdown("---")
            st.subheader("📝 OCR Extracted Text")
            st.write(" ".join(extracted_text))

            for brand in brands:
                if brand in extracted_text:
                    marque_detected = brand
                    break

            for etat in etat_options:
                if etat.upper() in extracted_text:
                    etat_detected = etat
                    break

            for text in extracted_text:
                if text.isdigit() and 2000 <= int(text) <= datetime.datetime.now().year:
                    annee_detected = int(text)
                    break

# ---- Prediction Functions ----
def preprocess_input(input_df):
    ohe_encoded = ohe.transform(input_df[['Marque', 'Type']])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(['Marque', 'Type']))
    ohe_df.index = input_df.index

    input_df['Etat_Encoded'] = ord_enc.transform(input_df[['Etat']])
    input_df_final = pd.concat([input_df, ohe_df], axis=1)
    input_df_final.drop(columns=['Marque', 'Type', 'Etat'], inplace=True)
    input_df_final = input_df_final.reindex(columns=feature_cols, fill_value=0)
    return input_df_final

def predict_price(input_df):
    processed = preprocess_input(input_df)
    dmatrix = xgb.DMatrix(processed)
    log_pred = xgb_model.predict(dmatrix)
    price = np.expm1(log_pred)
    return price[0]

# ---- Automatic Prediction ----
if marque_detected and mapped_type and etat_detected and annee_detected:
    auto_input_df = pd.DataFrame([{
        "Marque": marque_detected,
        "Etat": etat_detected,
        "Type": mapped_type,
        "Age": datetime.datetime.now().year - annee_detected
    }])

    try:
        auto_price = predict_price(auto_input_df)
        st.success(f"💰 Auto-estimated price based on detection: **{auto_price:,.2f} DH**")
    except Exception as e:
        st.error(f"Automatic prediction failed: {e}")

# ---- Manual Form Prediction ----
st.markdown("---")
with st.form("prediction_form"):
    st.markdown("Enter equipment details below to estimate its price:")

    marque = st.selectbox("Marque", brands, index=brands.index(marque_detected) if marque_detected else 0)
    type_options = ["bois", "driver", "serie", "hybride", "fers", "putter", "wedge"]
    type_ = st.selectbox("Type", type_options, index=type_options.index(mapped_type) if mapped_type in type_options else 0)
    etat = st.selectbox("État", etat_options, index=etat_options.index(etat_detected) if etat_detected else 0)
    annee = st.number_input("Année de sortie", min_value=2000, max_value=2025, value=annee_detected if annee_detected else 2022)
    age = datetime.datetime.now().year - annee

    submitted = st.form_submit_button("📈 Predict Price")
    if submitted:
        input_df = pd.DataFrame([{
            "Marque": marque,
            "Etat": etat,
            "Type": type_,
            "Age": age
        }])

        try:
            price = predict_price(input_df)
            st.success(f"💰 Estimated price: **{price:,.2f} DH**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
