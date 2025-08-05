# CLI LlamaIndex Chat with Local Files

Author: Harrison Forsyth — Applied Researcher in AI/XR

Welcome! My name is Harrison Forsyth. As an applied researcher focused on AI and XR (Extended Reality) development, I’m passionate about helping people tap into the power of modern language models and retrieval-augmented generation (RAG).

I've been doing a lot of work with RAG, and I thought this quick demo might be useful for anyone interested in seeing how AI can chat with their own documents.

This script lets you chat with any documents you drop into a folder, powered by local embeddings and an LLM via Ollama. It’s a lightweight example to kickstart your own explorations with Retrieval-Augmented Generation.**

Personal note: I tested this with just one document, but it should work for folders containing multiple files. Feel free to adapt and expand! Feedback and ideas are always welcome—let’s make AI easier to use for everyone.*

---

## Features

- **Embeddings:** Uses HuggingFace’s [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) for local, privacy-respecting vector embeddings.
- **LLM:** Connects to a local Ollama model (default: `llama3.2`). No API keys required!
- **Streaming & Verbose Stats:** See incremental answers and get optional timing/token usage info.
- **Flexible:** Just put TXT, PDF, Markdown, or other supported documents in a folder.

---

## Getting Started

### Prerequisites

- Python 3.8 or above
- [Ollama](https://ollama.com/) (running locally; with your desired model pulled, e.g., `ollama pull llama3`)

### Installation

1. Clone this repository, or copy the script `airag.py` into a folder.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure Ollama is running and the model name in the script (`llama3.2` by default) matches the available Ollama model.

### Usage

1. **Prepare some data**
    - Place your `.txt`, `.pdf`, `.md`, or similar files in the folder named `./data` (auto-created on first run).
    - I tested with one document in the folder; try that first!

2. **Run the chat:**

    ```bash
    python airag.py
    # For verbose logging:
    python airag.py --verbose
    ```

3. **Interact!**
    - Type your questions to the AI.
    - Use `exit` to quit.
    - Use `verbose on` and `verbose off` during chat to toggle output.

---

## Model & Data Notes

- **Embeddings:** Uses [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) by default.
- **LLM:** Uses local Ollama (default: `llama3.2`). To change models, update the `MODEL` variable in the script.
- **Document Loader:** Should work on txt, pdf, md, and more via [LlamaIndex](https://github.com/jerryjliu/llama_index).

---

## Troubleshooting

- On first run, if you see "Created directory 'data'...", just put your files there and re-run.
- Token usage stats may not be supported for all LLM backends or if streaming is limited.
- Make sure Ollama is running and the model is available.
- If you have just one document: it works! Try with more when you’re ready.

---

## Why?

Retrieval-augmented chat applications make it easy to turn your documents and knowledge bases into helpful, interactive bots—locally and privately. This is a simple launching point for prototyping your own RAG solutions, demos, or just tinkering with LlamaIndex and local LLMs.

---

## License & Feedback

MIT license.  
**Got feedback, questions, or want to collaborate on AI/XR or RAG projects? I’d love to hear from you!**

— Harrison Forsyth