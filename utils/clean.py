# cleaning the chunks
import re
import json
import os
from langchain_core.documents import Document

class Cleaner():
    
    def __init__(self):
        pass
    
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # document object -> str -> cleaned str
    def cleaned_document(self,document):
        content = "\n".join(doc.page_content for doc in document)
        cleaned_content = self.clean_text(content)
        return cleaned_content
    
    # after splitting
    def post_split(self,data):
        final_doc = []
        for entry in data:
            for i, doc in entry.items():
                doc = self.clean_text(doc)
                final_doc.append(doc)
        serialized_data = [
            {
                f"chunk {i}": text for i,text in enumerate(final_doc)
            }
        ]
        return serialized_data
    
    # before splitting
    def prepare_split(
        self,
        docs
    ):
        """Function to prepare a document for text splitter: returns a str to split"""  
        data = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in docs
        ]
        full_content = "\n\n".join([doc.page_content for doc in data])
        return full_content
