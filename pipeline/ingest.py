from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_fireworks import FireworksEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class Ingest():
    
    def __init__(
        self,
        collection_name,
        persist_history,
        document_loader=PyPDFDirectoryLoader,
        single_document_loader=PyPDFLoader,
        embeddings_model=FireworksEmbeddings,
        vector_store=Chroma,
        splitter=RecursiveCharacterTextSplitter,
        llm=HuggingFaceEndpoint,
        chat_model=ChatHuggingFace,
        prompt_template=PromptTemplate,
        output_parser=StrOutputParser,
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        search_type="mmr"
    ):
        """Default values
        
        document_loader=PyPDFDirectoryLoader,
        single_document_loader=PyPDFLoader,
        embeddings_model=FireworksEmbeddings,
        vector_store=Chroma,
        splitter=RecursiveCharacterTextSplitter,
        llm=HuggingFaceEndpoint,
        model=ChatHuggingFace,
        prompt_template=PromptTemplate,
        output_parser=StrOutputParser,
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        search_type="mmr"
        
        set values to change
        
        """
        self.document_loader = document_loader
        self.single_document_loader = single_document_loader
        self.embeddings_model = embeddings_model
        self.vector_store = vector_store
        self.splitter = splitter
        self.llm = llm
        self.chat_model = chat_model
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.repo_id = repo_id
        self.task = task
        self.search_type = search_type
        
        self.collection_name = collection_name
        self.persist_history = persist_history
        
        load_dotenv()
    
    # Loading document - folder
    def load_all_documents(
        self,
        file_path,
        
    ):
        """Function to load all documents, use when you have a directory of documents: returns a list"""
        loader = self.document_loader(
            path=file_path,
            glob="*.pdf"
        )

        docs = loader.lazy_load()
        return list(docs)
        
    # loading a single document
    def load_single_document(
        self,
        file_path,
        
    ):
        """Function to load a single document, use when you have only one file: returns a list"""
        loader = self.single_document_loader(
            file_path=file_path
        )
        return loader.load()

    # splitting of document
    def split_documents(
        self,
        content:str,
        chunk_size=700,
        chunk_overlap=20
    ):
        """Splits document using Recursive character splitter and returns list of dict: [chunk_id: text]"""

        splitted = self.splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = splitted.split_text(content)
        final_texts = [
            {
                f"chunk {i}": text for i, text in enumerate(texts)
            }
        ]
        return final_texts
    
    # embedding of chunks
    def embedd_chunks(
        self,
        embedding_model: str,
        data,
    ):
        """Embedds the chunks with default model"""
        embeddings = self.embeddings_model(
            model=embedding_model
        )
        final_doc = []
        for entry in data:
            for i, doc in entry.items():
                doc = Document(
                    page_content=doc,
                    metadata = {"chunk_id": i, "source": "Module 1"}
                )
                final_doc.append(doc)
                
        try:
            vector = self.vector_store.from_documents(
                embedding=embeddings,
                collection_name="module1",
                persist_directory="./CN_db",
                documents=final_doc
            )
            print(f"✅ Success! Embedded {len(final_doc)} chunks.")
            return True, vector
        except Exception as e:
            print(f"Final Debug Error: {e}")
            return False, None
        
    @staticmethod
    def clean_up_context(data):
        context = "\n".join(doc.page_content for doc in data)
        return context
        
    def get_response_final_chain(
        self,
        embedding_model: str,
        search_k:int, 
        search_lambda_mult:int,
    ):
        embedding = self.embeddings_model(
            model=embedding_model
        )

        vector = self.vector_store(
            embedding_function=embedding,
            collection_name=self.collection_name,
            persist_directory=self.persist_history,
        )

        retriever = vector.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                'k': search_k,
                'lambda_mult': search_lambda_mult
            }
        )
        # structure the LLM
        llm = self.llm(
            repo_id=self.repo_id,
            task=self.task
        )

        model = self.chat_model(llm=llm)

        prompt = self.prompt_template(
            template="""You are a Computer Networks genius. You have relevant data to a query related to computer networks. You will mention the module name (module 1: Introduction to Computer Networks), You will read the relevant context {context}, read the user's query {query}. Say 'I don't know' when the context and query do not match.""",
            input_variables=["context", "query"]
        )

        parser = self.output_parser()
        
        # retriever chain
        parallel_chain = RunnableParallel(
            {
                "context": retriever| RunnableLambda(self.clean_up_context),
                'query': RunnablePassthrough()
            }
        )

        chain = parallel_chain | prompt | model | parser
        return chain
