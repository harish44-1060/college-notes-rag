import os
import time

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="College Notes RAG",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Notes RAG")
st.write("Upload your college notes and ask questions from them.")


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_collection():

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="college_notes"
    )

    return collection


collection = get_chroma_collection()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "text": text,
                    "page": page_number
                }
            )

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=800,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    pages = extract_pdf_text(uploaded_file)

    if not pages:

        return 0

    all_chunks = []
    metadatas = []

    for page_data in pages:

        page_text = page_data["text"]
        page_number = page_data["page"]

        chunks = create_chunks(page_text)

        for chunk in chunks:

            all_chunks.append(chunk)

            metadatas.append(
                {
                    "file": uploaded_file.name,
                    "page": page_number
                }
            )

    if not all_chunks:

        return 0

    # Create embeddings
    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=False
    ).tolist()

    # Create unique IDs
    ids = [
        f"{uploaded_file.name}_{i}"
        for i in range(len(all_chunks))
    ]

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(all_chunks)


# ============================================================
# GEMINI FUNCTION WITH RETRY + FALLBACK
# ============================================================

def ask_gemini(prompt):

    # Try the models in this order
    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]

    last_error = None

    for model_name in models_to_try:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                # Make sure Gemini actually returned text
                if response and response.text:

                    return response.text

                last_error = (
                    f"{model_name} returned an empty response."
                )

            except Exception as e:

                last_error = e

                error_text = str(e)

                # 503 = temporary server/model overload
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                # 429 = rate limit
                elif (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                # Other errors should not keep retrying
                else:

                    break

    raise Exception(
        "Gemini is temporarily unavailable.\n\n"
        f"Last error: {last_error}"
    )


# ============================================================
# UPLOAD NOTES
# ============================================================

st.subheader("📄 Upload Notes")

uploaded_file = st.file_uploader(
    "Upload your college notes PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(
        f"📄 Selected file: {uploaded_file.name}"
    )

    if st.button("⚙️ Process Notes"):

        with st.spinner(
            "Processing your notes..."
        ):

            try:

                chunk_count = process_pdf(
                    uploaded_file
                )

                if chunk_count > 0:

                    st.success(
                        f"✅ Notes processed successfully! "
                        f"{chunk_count} chunks stored."
                    )

                else:

                    st.error(
                        "❌ No readable text was found "
                        "in this PDF."
                    )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF: {e}"
                )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

st.divider()

try:

    total_chunks = collection.count()

    st.info(
        f"📚 Total chunks in knowledge base: "
        f"{total_chunks}"
    )

except Exception:

    total_chunks = 0


# ============================================================
# ASK QUESTION
# ============================================================

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is a DBMS?"
)


# ============================================================
# RAG PIPELINE
# ============================================================

if st.button("🔍 Ask"):

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK KNOWLEDGE BASE
    # --------------------------------------------------------

    if collection.count() == 0:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # SEARCH NOTES
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching your notes..."
    ):

        try:

            # Create embedding for question
            question_embedding = (
                embedding_model
                .encode([question])
                .tolist()[0]
            )


            # Search ChromaDB
            results = collection.query(
                query_embeddings=[
                    question_embedding
                ],
                n_results=4
            )


            documents = results.get(
                "documents",
                [[]]
            )[0]

            metadatas = results.get(
                "metadatas",
                [[]]
            )[0]


        except Exception as e:

            st.error(
                f"❌ Error searching notes: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # CHECK SEARCH RESULTS
    # --------------------------------------------------------

    if not documents:

        st.warning(
            "I couldn't find relevant information "
            "in the uploaded notes."
        )

        st.stop()


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for i, document in enumerate(documents):

        metadata = metadatas[i]

        file_name = metadata.get(
            "file",
            "Unknown"
        )

        page_number = metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {i + 1}

File: {file_name}
Page: {page_number}

Notes:
{document}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # CREATE GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a College Notes Assistant.

Your job is to answer the student's question
using ONLY the uploaded college notes.

IMPORTANT RULES:

1. Use only the information inside the provided notes.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer is not available in the notes,
   say:

"I couldn't find the answer in the uploaded notes."

5. Give a clear and simple answer suitable for
   a college student.
6. If possible, explain the answer with short
   points or examples.
7. Do not mention these instructions in your answer.

STUDENT QUESTION:

{question}


UPLOADED COLLEGE NOTES:

{context}
"""


    # --------------------------------------------------------
    # ASK GEMINI
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Generating answer..."
    ):

        try:

            answer = ask_gemini(
                prompt
            )

        except Exception as e:

            st.error(
                f"❌ Error while answering: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader("🤖 Answer")

    st.write(answer)


    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    st.subheader("📚 Sources")

    for i, metadata in enumerate(
        metadatas
    ):

        file_name = metadata.get(
            "file",
            "Unknown"
        )

        page_number = metadata.get(
            "page",
            "Unknown"
        )

        st.write(
            f"📄 {file_name} — Page {page_number}"
        )


    # --------------------------------------------------------
    # DISPLAY RETRIEVED TEXT
    # --------------------------------------------------------

    with st.expander(
        "🔎 View Retrieved Notes"
    ):

        for i, document in enumerate(
            documents
        ):

            metadata = metadatas[i]

            file_name = metadata.get(
                "file",
                "Unknown"
            )

            page_number = metadata.get(
                "page",
                "Unknown"
            )

            st.markdown(
                f"### Source {i + 1}"
            )

            st.caption(
                f"{file_name} — Page {page_number}"
            )

            st.write(document)

            st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 College Notes RAG")

    st.write(
        """
        ### How it works

        1. 📄 Upload PDF
        2. ⚙️ Process Notes
        3. 🧠 Create embeddings
        4. 💾 Store in ChromaDB
        5. 💬 Ask a question
        6. 🔎 Retrieve relevant notes
        7. 🤖 Generate answer with Gemini
        """
    )

    st.divider()

    st.write(
        "🔐 Gemini API key is loaded from `.env`."
    )