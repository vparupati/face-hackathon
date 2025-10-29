import streamlit as st
import requests
from io import BytesIO
from PIL import Image

API = "http://127.0.0.1:8000"  # Default API URL

st.set_page_config(page_title="Face Hackathon Demo")

st.title("Face Hackathon Demo")
st.caption("Tabs: Similarity | Recognize | Expression. Start the FastAPI server first.")

tab1, tab2, tab3 = st.tabs(["Similarity", "Recognize", "Expression"])

with tab1:
    st.header("Face Similarity (Challenge 1)")
    c1, c2 = st.columns(2)
    img_a = c1.file_uploader("Image A", type=["jpg","jpeg","png","pgm"])
    img_b = c2.file_uploader("Image B", type=["jpg","jpeg","png","pgm"])
    
    # Display uploaded images side by side
    if img_a or img_b:
        st.subheader("Uploaded Images")
        col1, col2 = st.columns(2)
        with col1:
            if img_a:
                st.image(img_a, caption="Image A", width=200)
            else:
                st.write("No image uploaded")
        with col2:
            if img_b:
                st.image(img_b, caption="Image B", width=200)
            else:
                st.write("No image uploaded")
    
    if st.button("Compare", disabled=not(img_a and img_b)):
        files = {"image_a": img_a, "image_b": img_b}
        r = requests.post(f"{API}/similarity", files=files, timeout=120)
        if r.ok:
            res = r.json()
            st.metric("Score", f"{res['score']:.3f}", help="Higher = more similar")
            st.write(f"Verdict: **{res['verdict']}** (distance={res['distance']:.3f})")
        else:
            st.error(r.text)

with tab2:
    st.header("Recognize Person (Challenge 2)")
    img = st.file_uploader("Face image", type=["jpg","jpeg","png","pgm"], key="rec")
    
    # Display uploaded image
    if img:
        st.subheader("Uploaded Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Face to recognize", width=200)
    
    if st.button("Identify", disabled=not img):
        r = requests.post(f"{API}/recognize", files={"image": img}, timeout=120)
        if r.ok:
            res = r.json()
            if "error" in res:
                st.error(res["error"])
            else:
                st.metric("Predicted", res["label"])
                st.write(f"Confidence: {res['confidence']:.2f}")
        else:
            st.error(r.text)

with tab3:
    st.header("Expression (Challenge 3)")
    st.info("🎯 **ResNet18 RAF-DB Model v20251025** - Latest trained model")
    st.success("✅ **29.3% improvement** over C3 baseline!")
    st.caption("🔄 Using latest model: rafdb_expressions_v20251025_1431.onnx")
    img = st.file_uploader("Face image", type=["jpg","jpeg","png","pgm"], key="expr")
    
    # Display uploaded image
    if img:
        st.subheader("Uploaded Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Face for expression analysis", width=200)
    
    if st.button("Classify Expression", disabled=not img):
        r = requests.post(f"{API}/expression", files={"image": img}, timeout=120)
        if r.ok:
            res = r.json()
            if "error" in res:
                st.error(res["error"])
            else:
                st.metric("Expression", res["expression"])
                st.metric("Confidence", f"{res['confidence']:.3f}")
                if "model" in res:
                    st.caption(f"Model: {res['model']}")
                st.subheader("All Probabilities")
                st.json(res["probs"])
        else:
            st.error(r.text)
