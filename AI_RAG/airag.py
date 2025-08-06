import os
import time
import argparse

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# === Configurations ===
DATA_DIR = 'data'
MODEL = 'llama3.2'

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model=MODEL, request_timeout=360.0)
Settings.llm = llm

def build_index(data_dir):
    # Make sure data dir exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory '{data_dir}'. Put your files there and re-run this script.")
        exit(1)
    print(f"Indexing files in '{data_dir}' ...")
    docs = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    print(f"Indexed {len(docs)} files.")
    return index

def chat_loop(index, verbose=False):
    # Build a chat engine with streaming enabled
    try:
        chat_engine = index.as_chat_engine(llm=llm, chat_mode="context", stream=True)
    except TypeError:
        # Fallback for older versions
        chat_engine = index.as_chat_engine(llm=llm, chat_mode="context")

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

        # Try streaming the model's response, fallback to non-streaming
        try:
            response_stream = chat_engine.chat(user_input, stream=True)
        except TypeError:
            # If streaming is unsupported, fallback
            response_stream = [chat_engine.chat(user_input)]

        print("\nAI: ", end="", flush=True)
        full_response = ""
        for chunk in response_stream:
            # The chunk object/field may depend on your model; adapt as needed
            # chunk could be a string or a response object with `.response`
            text = getattr(chunk, "response", None) or getattr(chunk, "text", None) or str(chunk)
            print(text, end="", flush=True)
            full_response += text
        print("\n")  # AI output end

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
""")
    index = build_index(DATA_DIR)
    chat_loop(index, verbose=args.verbose)