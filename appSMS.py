import streamlit as st
import requests

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Classifier")
st.write("Enter your message below to check if it is **Spam** or **Ham (Not Spam)**.")

message = st.text_area("✍️ Write your message:")

if st.button("🔍 Predict"):
    if message.strip():
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json={"message": message})

            if response.status_code == 200:
                result = response.json()
                label = result["label"]
                prob = result["probability"]

                if label == "Spam":
                    st.error(f"🚨 Classified as **Spam** (Confidence: {prob:.2f})")
                else:
                    st.success(f"✅ Classified as **Ham** (Confidence: {prob:.2f})")
            else:
                st.error("❌ Error connecting to the API.")
        except Exception as e:
            st.error(f"⚠️ Exception: {e}")
    else:
        st.warning("⚠️ Please enter a message first.")
