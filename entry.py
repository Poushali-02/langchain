# using already created vectors here.
from pipeline.pipeline import Pipeline

pipeline = Pipeline(
    collection_name="module1",
    persist_history="CN_DB_2"
)

chain = pipeline.response_chain(
    embedding_model = "nomic-ai/nomic-embed-text-v1.5",
    search_k=3,
    search_lambda_mult=0.3
)

print(chain.invoke('Explain bus topology'))