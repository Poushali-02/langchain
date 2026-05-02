# NoteMe - Computer Networks RAG Application

A Retrieval-Augmented Generation (RAG) application for querying Computer Networks course material using LangChain, ChromaDB, and Llama 3.1.

## 📋 Overview

This application processes PDF documents containing Computer Networks lecture notes, creates vector embeddings, and enables intelligent question-answering through a conversational AI interface (coming soon). It uses a modular pipeline architecture for document ingestion, processing, and retrieval.

## 🏗️ Project Structure

```
noteme/
├── pipeline/             # Core RAG pipeline modules (multiple actions ingested together)
│   ├── ingest.py         # Document loading, splitting, and embedding (singular)
│   ├── pipeline.py       # Response chain for querying existing vectors (singular)
│   ├── chain.py          # Legacy chain implementation (singular)
│   ├── loader.py         # PDF loading utilities (singular)
│   ├── splitter.py       # Document splitting utilities (singular)
│   └── embed.py          # Vector embedding utilities (singular)
├── utils/                # Utility modules
│   └── clean.py          # Text cleaning and preprocessing
├── main.py               # Full pipeline: load → clean → embed → query
├── entry.py              # Quick query entry point (uses existing vectors)
├── .env                  # Environment variables (API keys)
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```

## 🔑 Key Components

### **pipeline/ingest.py**
The main ingestion class that handles:
- **Document Loading**: Loads PDF files from directories or single files
- **Text Splitting**: Chunks documents into manageable pieces (700 chars, 20 char overlap)
- **Vector Embedding**: Creates embeddings using Fireworks AI's Nomic model
- **Chain Creation**: Builds the complete RAG chain for question-answering

### **pipeline/pipeline.py**
Lightweight class for querying **existing** vector stores:
- Assumes embeddings are already created
- Focuses solely on retrieval and response generation
- Used in `entry.py` for quick queries

### **utils/clean.py**
Text preprocessing utilities:
- Removes excess whitespace
- Cleans raw document content before splitting
- Prepares text for optimal embedding

### **main.py**
Complete workflow demonstrating the full pipeline:
1. Load PDFs from `Computer networks/Module 1`
2. Clean the text
3. Split into chunks
4. Generate embeddings and store in ChromaDB
5. Create response chain and answer queries

### **entry.py**
Quick-start script for querying without re-processing:
- Uses pre-existing vectors from `CN_DB_2`
- Faster execution for repeated queries
- Ideal for testing and demos

## 🚀 Setup Instructions

### 1. **Prerequisites**
- Python 3.13+
- UV package manager (or pip)

### 2. **Install Dependencies**

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. **Environment Variables**

Create a `.env` file in the root directory:

```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**Get API Keys:**
- **Fireworks AI**: [https://fireworks.ai](https://fireworks.ai) (for embeddings)
- **Hugging Face**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (for Llama 3.1)

### 4. **Prepare Your Documents**

Place your PDF files in:
```
Computer networks/Module 1/
```

## 💻 Usage

### **Option 1: Full Pipeline (First Time)**

Run the complete ingestion pipeline:

```bash
python main.py
```

This will:
- Load all PDFs from the source directory
- Clean and split the text
- Create embeddings
- Store vectors in `CN_DB_2/`
- Answer the test query: "Explain bus topology"

### **Option 2: Query Existing Vectors**

For subsequent queries without re-processing:

```bash
python entry.py
```

This uses the pre-existing vector database and is much faster.

## 🔧 Configuration

### **Adjust Retrieval Parameters**

In `main.py` or `entry.py`, modify:

```python
chain = pipeline.response_chain(
    embedding_model="nomic-ai/nomic-embed-text-v1.5",
    search_k=3,              # Number of relevant chunks to retrieve
    search_lambda_mult=0.3   # MMR diversity (0=diverse, 1=similar)
)
```

### **Change Chunking Strategy**

In `ingest.py` → `split_documents()`:

```python
chunk_size=700,      # Characters per chunk
chunk_overlap=20     # Overlap between chunks
```

### **Switch LLM Model**

In `pipeline/ingest.py` or `pipeline/pipeline.py`:

```python
repo_id="meta-llama/Llama-3.1-8B-Instruct"  # Change to any HuggingFace model
```

## 📊 How It Works

1. **Document Loading**: PDFs are loaded using LangChain's `PyPDFLoader`
2. **Text Cleaning**: Removes excess whitespace and normalizes formatting
3. **Chunking**: Splits documents into 700-character chunks with 20-character overlap
4. **Embedding**: Each chunk is embedded using Nomic-embed-text-v1.5 (768 dimensions)
5. **Vector Storage**: Embeddings stored in ChromaDB with metadata
6. **Retrieval**: MMR (Maximal Marginal Relevance) search finds relevant chunks
7. **Generation**: Llama 3.1 generates answers based on retrieved context

## 🎯 Example Query

```python
chain.invoke('Explain bus topology')
```

**Output:**
```
Module 1: Introduction to Computer Networks

A bus topology is a multipoint network configuration where one long cable 
acts as a backbone to link all devices. All devices share a single 
communication channel...
```

## 📝 Customization

### **Add New Modules**

To process additional course modules:

```python
ingest = Ingest(
    collection_name="module2",      # New collection name
    persist_history="CN_DB_module2"  # Separate database
)
```

### **Custom Prompts**

Edit the prompt template in `pipeline/ingest.py`:

```python
template="""You are a Computer Networks expert. Context: {context}
Question: {query}. Provide a detailed technical answer."""
```

## 🐛 Troubleshooting

**Issue: "Fireworks API key not found"**
- Ensure `.env` file exists with `FIREWORKS_API_KEY`

**Issue: "ChromaDB collection already exists"**
- Delete `CN_DB_2/` directory to reset vectors
- Or use a different `collection_name`

**Issue: "PDF loading fails"**
- Verify PDFs are in `Computer networks/Module 1/`
- Check file permissions

## 📦 Dependencies

Key libraries:
- **LangChain**: RAG framework and orchestration
- **ChromaDB**: Vector database
- **Fireworks AI**: Embedding model provider
- **Hugging Face**: LLM hosting (Llama 3.1)
- **PyPDF**: PDF document parsing

## 🤝 Contributing

This is a learning project for Computer Networks course material. Feel free to adapt it for your own educational PDFs!

## 📄 License

Open source - adapt as needed for educational purposes.

---

**Built with LangChain 🦜🔗**