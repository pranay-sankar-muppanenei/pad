# Personal Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) chatbot built with LangChain,
FAISS, OpenAI embeddings, and Streamlit. It searches Markdown files in `docs/`
and answers questions through a browser-based chat interface.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this directory.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Create `.env` from `.env.example` and add your OpenAI API key:

   ```powershell
   Copy-Item .env.example .env
   ```

6. Start the frontend:

   ```powershell
   streamlit run streamlit_app.py
   ```

The browser UI will open at `http://localhost:8501`.

## How it works

1. Markdown files are loaded from `docs/`.
2. Documents are split into overlapping chunks.
3. OpenAI embeddings convert chunks into vectors.
4. FAISS retrieves the four most relevant chunks for a question.
5. An OpenAI chat model answers using only those retrieved chunks.

The original notebook, `rag_plus_langchain.ipynb`, remains available for
step-by-step experimentation.

## Before publishing

The sample documents contain personal, financial, health, travel, and contact
information. Replace or anonymize them before publishing a public repository.
Never commit `.env` or an API key.

To publish this as your own GitHub repository, create an empty repository on
GitHub, then run:

```powershell
git remote remove origin
git remote add origin https://github.com/<your-username>/<your-repository>.git
git add .
git commit -m "Build local RAG knowledge assistant"
git branch -M main
git push -u origin main
```
