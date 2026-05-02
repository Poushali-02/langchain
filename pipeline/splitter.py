# embeddding the first module
import json
import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..\\docs", "mod1loader.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    docs = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in data
    ]

full_content = "\n\n".join([doc.page_content for doc in docs])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=20
)
texts = splitter.split_text(full_content)

json_output_path = os.path.join(base_dir, "..\\docs", "mod1splitted_2.json")

final_texts = [
    {
        f"chunk {i}": text for i, text in enumerate(texts)
    }
]

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(final_texts, f, ensure_ascii=False, indent=4)