from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_fireworks import FireworksEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings_model = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1.5",
)

vector_store = Chroma(
    embedding_function=embeddings_model,
    collection_name="module1",
    persist_directory="./CN_db",
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 3,
        'lambda_mult': 0.2
    }
)

# call the LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""You are a Computer Networks genius. You have relevant data to a query related to computer networks. You will mention the module name (module 1: Introduction to Computer Networks), You will read the relevant context {context}, read the user's query {query}. Say 'I don't know' when the context and query do not match.""",
    input_variables=["context", "query"]
)

parser = StrOutputParser()

def clean_up_context(data):
    context = "\n".join(doc.page_content for doc in data)
    return context

# retriever chain
parallel_chain = RunnableParallel(
    {
        "context": retriever| RunnableLambda(clean_up_context),
        'query': RunnablePassthrough()
    }
)

chain = parallel_chain | prompt | model | parser