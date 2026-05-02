import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import json

load_dotenv()

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "Computer networks/Module 1")

loader = PyPDFDirectoryLoader(
    path=file_path,
    glob="*.pdf"
)

docs = loader.lazy_load()

all_docs = list(docs)

json_output_path = os.path.join(base_dir, "..", "processed_docs.json")

serializable_docs = [
    {
        "page_content": doc.page_content, 
        "metadata": doc.metadata
    } 
    for doc in all_docs
]

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(serializable_docs, f, ensure_ascii=False, indent=4)

print(f"Successfully saved all documents to: {json_output_path}")