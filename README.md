# SmartStudy AI Pro

A Streamlit study assistant that extracts text from course materials, indexes it in a local ChromaDB collection, and uses retrieval-augmented generation (RAG) for document-grounded chat, summaries, quizzes, and flashcards.

## Features

- Upload PDF, DOCX, TXT, JPG, JPEG, and PNG files (20 MB maximum per file).
- Extract PDF and DOCX text; run OCR on images and scanned PDF pages with Tesseract.
- Store workspace-scoped document chunks and embeddings in persistent local ChromaDB; reopen the bookmarked workspace URL to restore documents.
- Ask questions with retrieved source snippets and page references where available.
- Generate summaries, validated four-option MCQs, scored quizzes, and navigable flashcards.
- Select Gemini or OpenAI separately for generation and embeddings.

## Requirements

- Python 3.10 or newer.
- A Gemini or OpenAI API key for each selected AI provider.
- Tesseract OCR installed and available on `PATH` to read images or scanned PDF pages.

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `LLM_PROVIDER` and `EMBEDDING_PROVIDER` to `gemini` or `openai`. Add the matching credentials as `GOOGLE_API_KEY` and/or `OPENAI_API_KEY`. Keys are read from the environment and never shown in the UI.

To use Tesseract on Windows, install the Tesseract OCR application and ensure its executable is on `PATH`. The app reports an actionable error when OCR is unavailable.

## Run

```powershell
streamlit run app.py
```

The selected provider credentials are needed to index documents and use AI study tools. The landing page and document extraction remain available without them.

## Configuration

| Variable | Values | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini`, `openai` | Text generation |
| `EMBEDDING_PROVIDER` | `gemini`, `openai` | Document and query embeddings |
| `GOOGLE_API_KEY` | Secret key | Gemini access |
| `OPENAI_API_KEY` | Secret key | OpenAI access |

Do not commit `.env`, uploaded documents, or `data/chroma_db`. Gemini uses `gemini-2.5-flash` and `gemini-embedding-001`; OpenAI uses `gpt-4o-mini` and `text-embedding-3-small`.

## Privacy and storage

Extracted text is processed by the app and vector embeddings are stored under `data/chroma_db/`. The vector collection is workspace-filtered, and document contents are sent to the configured AI provider when you ask questions or request a study aid. Chat history and dashboard statistics are session-only. Removing a document deletes its stored vectors for the current workspace. The workspace ID in the page URL is a private access link; keep it private. The app has no authentication and is intended for a local demonstration, not shared public deployment.

## Tests

Run the offline core tests:

```powershell
python -m unittest discover -s tests -v
```

The tests cover text extraction, upload validation, chunk overlap, and quiz scoring. Provider integration requires valid API credentials and is not exercised by the offline tests.
# smart-study-ai-pro
# smart-study-ai-pro
# smart-study-ai-pro
# smart-study-ai-pro
# smart-study-ai-pro
# smart-study-ai-pro
