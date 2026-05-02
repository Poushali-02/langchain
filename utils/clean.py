# cleaning the chunks
import re
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "docs", "mod1splitted_2.json")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    
final_doc = []
for entry in data:
    for i, doc in entry.items():
        doc = clean_text(doc)
        final_doc.append(doc)

json_output_path = os.path.join(base_dir, "docs", "mod1_cleaned_docs_2.json")

serialized_data = [
    {
        f"chunk {i}": text for i,text in enumerate(final_doc)
    }
]

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(serialized_data, f, ensure_ascii=False, indent=4)
    
print(f"Successfully saved all documents to: {json_output_path}")