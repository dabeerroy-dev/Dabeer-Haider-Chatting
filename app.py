import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import tempfile

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
    }
    
    /* Main container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.3);
        margin-top: 20px;
    }
    
    /* Title */
    h1 {
        background: linear-gradient(90deg, #ffffff, #f0e6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* User message bubble */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 14px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0 10px 60px;
        color: white;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* AI message bubble */
    .ai-message {
        background: rgba(255,255,255,0.95);
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 60px 10px 0;
        color: #333;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Upload box */
    .stFileUploader {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 15px !important;
        padding: 10px !important;
        border: 2px dashed rgba(255,255,255,0.5) !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(0,255,150,0.2) !important;
        border: 1px solid rgba(0,255,150,0.4) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Chat input */
    .stChatInput {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 25px !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.3) !important;
    }

    /* Spinner */
    .stSpinner {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Beautiful Header
# ============================================
st.markdown("""
<div style='text-align:center; padding: 10px 0 20px 0;'>
    <div style='font-size: 4rem;'>🧠</div>
    <h1 style='background: linear-gradient(90deg, #fff, #f0e6ff);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-size: 2.5rem;
               font-weight: 800;'>
        AI PDF Chatbot
    </h1>
    <p style='color: rgba(255,255,255,0.85); font-size: 1.1rem;'>
        ✨ Upload any PDF and chat with it instantly!
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Groq setup
import os
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Upload
st.markdown(
    "<p style='color:white; font-weight:600;'>📄 Upload Your PDF:</p>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    label_visibility="collapsed"
)

if uploaded_file is not None:

    with st.spinner("✨ Reading your PDF..."):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("pdf_docs")
        except:
            pass
        collection = client.create_collection("pdf_docs")

        texts = [c.page_content for c in chunks
                 if c.page_content.strip() != ""]

        if texts:
            embeddings = model.encode(texts).tolist()
            collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=[f"chunk_{i}" for i in range(len(texts))]
            )

    st.success(f"✅ {uploaded_file.name} ready! Ask me anything!")
    st.divider()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 &nbsp;{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="ai-message">🤖 &nbsp;{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    # Chat input
    question = st.chat_input("✨ Ask anything about your PDF...")

    if question:
        st.markdown(
            f'<div class="user-message">👤 &nbsp;{question}</div>',
            unsafe_allow_html=True
        )
        st.session_state.messages.append({
            "role": "user", "content": question
        })

        with st.spinner("🤔 Thinking..."):
            q_emb = model.encode([question]).tolist()
            results = collection.query(
                query_embeddings=q_emb,
                n_results=3
            )
            context = "\n\n".join(results["documents"][0])

            prompt = f"""You are a helpful assistant.
Answer using ONLY the context below.
If answer not in context, say 'I dont know'.
Context: {context}
Question: {question}
Answer:"""

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content

        st.markdown(
            f'<div class="ai-message">🤖 &nbsp;{answer}</div>',
            unsafe_allow_html=True
        )
        st.session_state.messages.append({
            "role": "assistant", "content": answer
        })