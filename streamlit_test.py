# ----------------- Import dan Setup -----------------
# import os
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# os.environ["ACCELERATE_DISABLE_RICH"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import torch # Import torch untuk penanganan device yang lebih eksplisit

# ----------------- Model & FAISS Setup -----------------

@st.cache_resource
def load_model():
    # Pastikan hanya pakai PyTorch dan set device secara eksplisit
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

model = load_model()

## FAISS & Cosine
def build_faiss_index_cosine(texts):
    # 1. Buat embedding
    embeddings = model.encode(texts, convert_to_numpy=True)
 
    # 2. Normalisasi agar inner product = cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')  # FAISS hanya menerima float32
 
    # 3. Buat index FAISS dengan inner product
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
 
    return index, embeddings

## Retrieval
def retrieve(query, index, df, top_k=None):
    return df  
 
## LLM - Generate Answer
def generate_answer(query, context, api_key):
    openai.api_key = api_key
    system_message = "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data yang diberikan."
    user_message = f"""
    Pertanyaan: {query}
 
    Data yang relevan:
    {context}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0]['message']["content"].strip()
 
def transform_data(df, selected_columns):
    df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
    return df   

# ----------------- UI Streamlit -----------------

st.title("🔍 Question Answering from CSV using LLM & FAISS")

st.sidebar.header("📂 Upload CSV dan API Key")

uploaded_file = st.sidebar.file_uploader("Upload File CSV", type='csv')
input_api_key = st.sidebar.text_input("🔑 Masukan API Key OpenAI", type='password')
button_api = st.sidebar.button('Aktifkan API Key')

if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'history' not in st.session_state:
    st.session_state.history = []

if input_api_key and button_api:
    st.session_state.api_key = input_api_key
    st.sidebar.success("✅ API Key Aktif")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📊 Pilih Kolom untuk Analisis")
    selected_columns = st.multiselect(
        "Pilih Kolom:",
        options=df_raw.columns.to_list(),
        default=df_raw.columns.to_list()
    )

    if not selected_columns:
        st.warning("⚠️ Harap pilih setidaknya satu kolom.")
        st.stop()

    st.dataframe(df_raw[selected_columns])

    query = st.text_input("❓ Masukan Pertanyaan Anda")
    run_query = st.button("🔎 Jawab Pertanyaan")

    if run_query:
        try:
            if not st.session_state.api_key:
                st.warning("🔐 Anda harus mengaktifkan API Key terlebih dahulu.")
                st.stop()

            df_transformed = transform_data(df_raw.copy(), selected_columns)
            if "text" not in df_transformed.columns or df_transformed["text"].empty:
                st.error("Tidak ada data yang valid untuk dianalisis setelah transformasi kolom. Pastikan kolom yang dipilih berisi data.")
                st.stop()

            if not df_transformed['text'].to_list():
                st.warning("Tidak ada data teks yang ditemukan untuk membangun indeks FAISS.")
                st.stop()

            index, _ = build_faiss_index_cosine(df_transformed['text'].to_list())

            if index is None:
                st.error("Gagal membangun indeks FAISS. Mungkin tidak ada data yang valid.")
                st.stop()

            with st.spinner("🔍 Mencari data relevan..."):
                results = retrieve(query, index, df_transformed)
                if results.empty:
                    context = ""
                else:
                    context = "\n".join(results["text"].to_list())

                if not context:
                    st.warning("Tidak ada konteks yang relevan ditemukan untuk pertanyaan Anda.")
                    answer = "Tidak ada informasi yang relevan ditemukan dalam data yang disediakan untuk menjawab pertanyaan Anda."
                    st.subheader("💬 Jawaban:")
                    st.info(answer)
                    st.session_state.history.append((query, answer))
                    st.stop()

            with st.spinner("💬 Menghasilkan jawaban..."):
                answer = generate_answer(query, context, st.session_state.api_key)

            st.subheader("💬 Jawaban:")
            st.success(answer)
            st.session_state.history.append((query, answer))
        except ValueError as ve:
            st.error(f"❌ Error: {str(ve)}")
        except Exception as e:
            st.error(f"❌ Terjadi error tak terduga: {str(e)}")
            st.exception(e)
    elif run_query and not st.session_state.api_key:
        st.warning("🔐 Anda harus mengaktifkan API Key terlebih dahulu.")
else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")

# ----------------- HISTORY -----------------
if st.session_state.history:
    st.subheader("⏰ Riwayat Pertanyaan dan Jawaban")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"❓ Pertanyaan #{i}: {q}"):
            st.markdown(f"💭 **Jawaban:** {a}")
