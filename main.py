import os
from pipeline.ingest import Ingest
from utils.clean import Cleaner
from langchain_core.documents import Document

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Computer networks/Module 1")

cleaner = Cleaner()

ingest = Ingest(
    collection_name="module1",
    persist_history="CN_DB_2"
)

# load documents
docs = ingest.load_all_documents(file_path=file_path)
print("✅ Documents loaded")

# clean the document
cleaned_doc = cleaner.cleaned_document(docs)
print("✅ Documents cleaned")

# split the document
splitted = ingest.split_documents(
    content=cleaned_doc
)
print("✅ Documents splitted")

# embedd the splitted chunks
vector = ingest.embedd_chunks(
    embedding_model="nomic-ai/nomic-embed-text-v1.5",
    data=splitted
)
print("✅ Vectors generated")

chain = ingest.get_response_final_chain(
    embedding_model = "nomic-ai/nomic-embed-text-v1.5",
    search_k=3,
    search_lambda_mult=0.3
)
print("✅ Chain created")

print(chain.invoke('Explain bus topology'))