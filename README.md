# 🎓 College Notes RAG

College Notes RAG is a Retrieval-Augmented Generation application that helps students ask questions from their college notes stored in PDF files.

## Features

- Upload college notes as PDF
- Extract and split PDF text into chunks
- Generate embeddings using Sentence Transformers
- Store and retrieve notes using ChromaDB
- Generate answers using Google Gemini
- Display the source file and page number

## Technologies

Python | Streamlit | PyPDF | Sentence Transformers | ChromaDB | Google Gemini

## How It Works

PDF Notes → Text Extraction → Chunking → Embeddings → ChromaDB → Question → Similarity Search → Gemini → Answer

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py


## 🔑 Setup
Add your Gemini API key to a `.env` file as `GOOGLE_API_KEY`.


Objective
To provide a simple AI-based assistant that allows students to quickly find answers from their college notes using RAG.

GitHub
https://github.com/harish44-1060/college-notes-rag