from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import json
import os
from langchain_fireworks import FireworksEmbeddings

load_dotenv()

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..\\docs", "mod1_cleaned_docs_2.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    
final_doc = []
for entry in data:
    for i, doc in entry.items():
        doc = Document(
            page_content=doc,
            metadata = {"chunk_id": i, "source": "Module 1"}
        )
        final_doc.append(doc)
        
embeddings_model = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1.5",
)

try:
    vector_store = Chroma.from_documents(
        embedding=embeddings_model,
        collection_name="module1",
        persist_directory="./CN_db",
        documents=final_doc
    )
    print(f"✅ Success! Embedded {len(final_doc)} chunks.")
except Exception as e:
    print(f"Final Debug Error: {e}")