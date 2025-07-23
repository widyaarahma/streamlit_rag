# ----------------- Import dan Setup -----------------
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import torch



# ----------------- Model & FAISS Setup -----------------
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

model = load_model()

def build_faiss_index_cosine(texts):
    if not texts:
        return None, None
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(query, index, df, top_k=5):
    if index is None:
        return pd.DataFrame()
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.astype('float32')

    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    actual_top_k = min(top_k, index.ntotal)
    if actual_top_k == 0:
        return pd.DataFrame()

    D, I = index.search(query_embedding, actual_top_k)
    return df.iloc[I[0]]

def generate_answer(query, context, api_key):
    if not api_key:
        raise ValueError("API Key OpenAI belum diatur.")
    openai.api_key = api_key
    system_message = "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data yang diberikan. Jawablah dengan ringkas dan fokus pada informasi yang relevan dari data yang disediakan. Jika informasi tidak ditemukan dalam data, nyatakan bahwa Anda tidak dapat menjawab pertanyaan berdasarkan data yang ada."
    user_message = f"""
    Pertanyaan: {query}

    Data yang relevan:
    {context}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0]['message']["content"].strip()
    except openai.error.AuthenticationError:
        raise ValueError("API Key OpenAI tidak valid.")
    except openai.error.RateLimitError:
        raise ValueError("Batas rate OpenAI API terlampaui.")
    except openai.error.OpenAIError as e:
        raise ValueError(f"Terjadi error dari OpenAI API: {e}")
    except Exception as e:
        raise ValueError(f"Terjadi error tidak terduga: {e}")

def transform_data(df, selected_columns):
    valid_columns = [col for col in selected_columns if col in df.columns]
    if not valid_columns:
        st.warning("Tidak ada kolom yang dipilih ditemukan dalam file CSV.")
        return pd.DataFrame(columns=["text"])

    df["text"] = df[valid_columns].astype(str).agg(" | ".join, axis=1)
    return df

# ----------------- UI Streamlit -----------------
st.title("🔍 Question Answering from CSV using LLM & FAISS")

# Inisialisasi session_state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.header("📂 Upload CSV dan API Key")

uploaded_file = st.sidebar.file_uploader("Upload File CSV", type='csv')
input_api_key = st.sidebar.text_input("🔑 Masukan API Key OpenAI", type='password')
button_api = st.sidebar.button('Aktifkan API Key')

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
                st.error("Data tidak valid setelah transformasi.")
                st.stop()

            if not df_transformed['text'].to_list():
                st.warning("Tidak ada data teks untuk membangun indeks FAISS.")
                st.stop()

            index, _ = build_faiss_index_cosine(df_transformed['text'].to_list())

            if index is None:
                st.error("Gagal membangun indeks FAISS.")
                st.stop()

            with st.spinner("🔍 Mencari data relevan..."):
                results = retrieve(query, index, df_transformed)
                context = "\n".join(results["text"].to_list()) if not results.empty else ""

                if not context:
                    answer = "Tidak ada informasi yang relevan ditemukan dalam data yang disediakan untuk menjawab pertanyaan Anda."
                    st.subheader("💬 Jawaban:")
                    st.info(answer)
                    st.session_state.history.append((query, answer))
                    st.stop()

            with st.spinner("💬 Menghasilkan jawaban..."):
                answer = generate_answer(query, context, st.session_state.api_key)

            st.subheader("💬 Jawaban:")
            st.success(answer)

            
            # Simpan ke riwayat
            st.session_state.history.append((query, answer))

        except ValueError as ve:
            st.error(f"❌ Error: {str(ve)}")
        except Exception as e:
            st.error(f"❌ Terjadi error tak terduga: {str(e)}")
            st.exception(e)

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")

# ----------------- HISTORY -----------------
if st.session_state.history:
    st.subheader("⏰ Riwayat Pertanyaan dan Jawaban")
    for i, (q, a) in enumerate(reversed(st.session_state.history[-5:]), 1):
        with st.expander(f"❓ Pertanyaan #{i}: {q}"):
            st.markdown(f"💭 **Jawaban:** {a}")
