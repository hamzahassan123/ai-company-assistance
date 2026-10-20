# Temp Project

A command-line retrieval-augmented question-answering app backed by a local
FAISS index and Gemini.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A Google Gemini API key

## Setup

From the project directory, create `.env` with:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Install the project and its dependencies:

```powershell
uv sync
```

## Run

Ask a question using either command:

```powershell
uv run temp-project
```

```powershell
uv run python -m temp_project.main
```

The repository includes a vector database in `data/vectorDB`. To rebuild it
from the Markdown files in `data/sales`, run:

```powershell
uv run temp-project-ingest
```

## Evaluate

Configure a LangSmith API key and create the `cogent-mind-evaluation` dataset
before running:

```powershell
uv run python -m temp_project.evaluate
```