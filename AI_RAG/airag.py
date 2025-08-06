import warnings

# Suppress FutureWarnings and UserWarnings (especially from torch/HF)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*encoder_attention_mask.*BertSdpaSelfAttention.*",
    category=FutureWarning
)

import os
import sys
import time
import argparse

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# === Configurations ===
DATA_DIR = 'data'
PERSIST_DIR = 'index_store'               # Where the index is persisted
MODEL = 'llama3'                          # Change if your Ollama model differs

SYSTEM_PROMPT = (
    "You are a helpful, conversational AI assistant. "
    "Only reference the user's files and documents if the user asks about them. "
    "Otherwise, focus on casual, friendly, open-ended conversation."
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model=MODEL, request_timeout=360.0)
Settings.llm = llm

def build_or_load_index(data_dir, persist_dir):
    """Load a persisted index if available, otherwise build and persist it."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory '{data_dir}'. Put your files there and re-run this script.")
        sys.exit(1)

    if os.path.exists(persist_dir):
        try:
            print("Loading existing index from disk...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("Index loaded from disk.")
            return index
        except Exception as e:
            print(f"Failed to load index from {persist_dir}: {e}")
            print("Rebuilding index from scratch.")

    print(f"Indexing files in '{data_dir}' ...")
    docs = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Indexed {len(docs)} files and saved to '{persist_dir}'.")
    return index

def chat_loop(index, verbose=False):
    # Build a chat engine with streaming and a system prompt (if supported)
    try:
        chat_engine = index.as_chat_engine(
            llm=llm,
            chat_mode="context",
            stream=True,
            system_prompt=SYSTEM_PROMPT
        )
    except TypeError:
        # Fallback: if system_prompt kwarg missing, try other alternatives, or run without it
        try:
            chat_engine = index.as_chat_engine(
                llm=llm,
                chat_mode="context",
                stream=True,
                system_message=SYSTEM_PROMPT
            )
        except TypeError:
            chat_engine = index.as_chat_engine(
                llm=llm,
                chat_mode="context",
                stream=True
            )

    print("\nChat session started! Type 'exit' to quit. Type 'verbose on' / 'verbose off' to toggle stats.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "verbose on":
            verbose = True
            print("[Verbose mode enabled]")
            continue
        if user_input.lower() == "verbose off":
            verbose = False
            print("[Verbose mode disabled]")
            continue

        t0 = time.time()
        try:
            response_stream = chat_engine.chat(user_input, stream=True)
        except TypeError:
            response_stream = [chat_engine.chat(user_input)]

        print("\nAI: ", end="", flush=True)
        full_response = ""
        for chunk in response_stream:
            text = getattr(chunk, "response", None) or getattr(chunk, "text", None) or str(chunk)
            print(text, end="", flush=True)
            full_response += text
        print("\n")

        t1 = time.time()
        elapsed = t1 - t0

        if verbose:
            print(f"- Elapsed: {elapsed:.02f} sec")
            try:
                tokens = getattr(chunk, "usage", {}).get("total_tokens") or getattr(chunk, "token_count", None)
            except Exception:
                tokens = None
            if tokens is not None:
                print(f"- Tokens: {tokens}")
                if isinstance(tokens, (int, float)) and tokens:
                    print(f"- Tokens/sec: {tokens / elapsed:.2f}")
            else:
                print("- [Token count/statistics unavailable in streaming mode]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CLI Chat Demo (LLM-powered, streaming output)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose LLM output/stats")
    args = parser.parse_args()
    print("""
==== CLI Chat Demo ====
Drop files (txt, pdf, md, etc) into the './data' folder.

* The vector index will be loaded from disk if available,
  otherwise it will be built and persisted for future runs.
""")
    index = build_or_load_index(DATA_DIR, PERSIST_DIR)
    chat_loop(index, verbose=args.verbose)