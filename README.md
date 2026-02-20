# RAG Financial Analyst 📈🤖

An advanced Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, and **Streamlit** to analyze complex 10-K financial reports. This application allows users to interact conversationally with financial documents from major tech companies (AMD, Intel, NVIDIA) using a localized large language model (LLM) for maximum privacy and security.

---

## 🌟 Key Features

- **Intelligent Document Retrieval:** Utilizes **ChromaDB** and the high-performance `BAAI/bge-m3` embedding model for accurate, dense vector retrieval of financial data.
- **Contextual Conversation Memory:** Implements a sliding window `MemoryChain` that reformulates user queries based on chat history to ensure highly relevant answers during multi-turn conversations.
- **Company-Specific Agents:** Features distinct semantic prompts and retrieval chains tailored independently for AMD, Intel, and NVIDIA financial analysis.
- **Local Model Execution:** Integrates seamlessly with local/self-hosted LLMs using **LM Studio** (OpenAI API compliant), ensuring sensitive financial queries never leave your machine.
- **Modern User Interface:** A responsive, premium-styled chat interface built with **Streamlit**.
- **Containerized Deployment:** Fully Dockerized architecture via `docker-compose` for rapid deployment and reproducibility.

---

## 🛠️ Technology Stack

- **Backend / Orchestration:** Python 3.11, LangChain
- **Frontend / UI:** Streamlit
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`BAAI/bge-m3`)
- **LLM Interface:** LM Studio (Local deployment)
- **Document Processing:** PyMuPDF, Langchain Text Splitters
- **Infrastructure:** Docker, Docker Compose

---

## 📁 Project Structure

```text
rag_system/
├── documents/            # Source 10-K PDF financial reports (organized by company)
│   ├── amd/
│   ├── intel/
│   └── nvidia/
├── src/                  # Application source code
│   ├── agents.py         # LangChain logic, MemoryChain, and Prompts
│   ├── app.py            # Streamlit web application frontend
│   ├── config.py         # Environment variables and path configs
│   └── vector_store.py   # Vector DB initialization and retriever functions
├── vector_dbs/           # Persisted ChromaDB vector stores (auto-generated)
├── .env                  # Environment Variables (Optional)
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker services definition
├── rag.ipynb             # Document ingestion and vector database creation script
└── requirements.txt      # Python dependencies
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

- **Python 3.11+** (if running locally without Docker)
- **Docker & Docker Compose**
- **LM Studio** installed locally.

### 2. Prepare the Documents & Vector Database

Before launching the chat interface, you must ingest the financial PDFs and generate the vector embeddings.

1. Place your 10-K PDF files into their respective folders inside `documents/` (e.g., `documents/amd`, `documents/intel`, `documents/nvidia`).
2. Install the required local dependencies to run the notebook:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the `rag.ipynb` Jupyter Notebook. This script will:
   - Load the PDFs using `PyMuPDFLoader`.
   - Chunk the documents using `RecursiveCharacterTextSplitter`.
   - Generate embeddings using `BAAI/bge-m3`.
   - Persist the Chroma vector databases into the `vector_dbs/` directory.

### 3. Start the Local LLM (LM Studio)

1. Open **LM Studio**.
2. Download and load your preferred model (e.g., a `Qwen` or `Llama 3` model optimized for instructions).
3. Start the **Local Inference Server**.
4. Ensure it is running on the default port: `http://localhost:1234/v1`.

### 4. Run the Application (Docker)

The easiest way to run the Streamlit frontend and connect it to your Vector DBs and LM Studio is via Docker Compose.

```bash
# Build and start the container in detached mode
docker-compose up --build -d
```

> **Note on LM Studio connection:** The `docker-compose.yml` uses `host.docker.internal` so the Docker container can communicate with LM Studio running on your host machine.

### 5. Access the Web Interface

Once the container is running, open your web browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🧠 System Architecture & Strategies

### Strategy 1: Data Ingestion & Chunking

Financial reports are dense and context-heavy. In `rag.ipynb`, files are loaded and split into chunks of `512` characters with an overlap of `256` characters. The overlap is crucial for maintaining sentences and financial clauses that might span across chunk boundaries.

### Strategy 2: Modularized RAG Chain

The backend is split into distinct modules:

- `vector_store.py`: Centralizes the embedding model instantiation (using `CPU` for standard inference) and retrieves the persisted Chroma stores.
- `agents.py`: Implements a custom `MemoryChain`. Instead of passing the entire chat history blindly to the LLM, the system uses a **History-Aware Retriever**. It reformulates the user's latest question using the past 5 interactions, turning context-dependent queries (e.g., "How much did it grow in Q2?") into standalone queries (e.g., "How much did AMD's revenue grow in Q2 of 2023?").

### Strategy 3: Company-Specific System Prompts

To prevent hallucinations and cross-contamination of data, each company agent (AMD, Intel, NVIDIA) is instantiated with a highly specific `ChatPromptTemplate`. The LLM is strictly instructed to act as a financial analyst for that specific company and to output numbers explicitly.

### Strategy 4: Streamlit Session State Management

The UI in `app.py` actively manages `st.session_state` to decouple memory between agents. If a user switches from the AMD agent to the Intel agent, the Streamlit session state instantly clears the chat and re-instantiates the `MemoryChain` to prevent the LLM from inadvertently answering Intel questions with AMD context.
