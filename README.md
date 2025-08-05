# NeuroSync

**AI-Native ETL Pipeline for RAG and LLM Applications**

NeuroSync is a comprehensive, production-ready ETL pipeline specifically designed for Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) applications. Build powerful AI applications from any data source with a single command.

## Quick Start

```bash
# Install NeuroSync
pip install neurosync

# Process any data source with one command
neurosync run /path/to/your/data

# That's it! NeuroSync handles:
# Data ingestion from any source
# Intelligent processing and chunking
# Embedding generation
# Vector store creation
# LLM setup and interactive chat
```

## 🎯 Why NeuroSync?

### **One Command, Complete Pipeline**
```bash
# Traditional approach (multiple tools, complex setup)
# Step 1: Write ingestion scripts
# Step 2: Set up processing pipeline
# Step 3: Configure embedding service
# Step 4: Set up vector database
# Step 5: Integrate LLM providers
# Step 6: Build chat interface

# NeuroSync approach (one command)
neurosync run /your/data --auto
# 🎉 Done! Chat-ready AI application in minutes
```

### **Intelligent Auto-Detection**
NeuroSync automatically detects your data type and selects optimal configurations:

- **📁 Files/Directories** → Intelligent file processing with format detection
- **APIs** → Rate-limited ingestion with authentication support
- **Databases** → Optimized queries for PostgreSQL, MySQL, SQLite
- **Cloud Storage** → S3, GCS, Azure Blob integration

### **Production-Ready from Day One**
- **Auto-scaling**: Handles datasets from KB to TB
- **Error Resilience**: Robust error handling and recovery
- **Monitoring**: Real-time metrics and performance tracking
- **Security**: Encryption, authentication, and audit logging

## Key Features

### **1. Universal Data Ingestion**
```bash
# Files and directories
neurosync run /docs/technical-manuals/

# REST APIs
neurosync run https://api.company.com/knowledge-base

# Databases
neurosync run postgresql://user:pass@host:5432/knowledge_db

# Multiple sources
neurosync run config/multi-source.yaml
```

### **2. Intelligent Processing**
- **Semantic Chunking**: Preserves meaning across chunk boundaries
- **Quality Scoring**: AI-powered content quality assessment
- **Deduplication**: Remove redundant content automatically
- **Language Detection**: Multi-language support with optimization

### **3. Advanced Embeddings**
- **HuggingFace Models**: 100+ pre-trained models
- **OpenAI Integration**: Latest embedding models (ada-002, text-embedding-3)
- **⚡ Batch Processing**: Optimized for high-throughput
- **💾 Smart Caching**: Avoid recomputing existing embeddings

### **4. Scalable Vector Storage**
- **FAISS Integration**: Multiple index types (Flat, HNSW, IVF)
- **Qdrant Support**: Production vector database
- **Hybrid Search**: Combine dense and sparse retrieval
- **Versioning**: Built-in backup and rollback capabilities

### **5. Multi-LLM Integration**
- **OpenAI**: GPT-3.5, GPT-4, GPT-4o
- **Anthropic**: Claude 3 (Haiku, Sonnet, Opus)
- **Cohere**: Command and Chat models
- **Google**: Gemini Pro and Ultra
- **Auto-Fallback**: Seamless provider switching

## Template System

NeuroSync uses intelligent templates that adapt to your needs:

### **Processing Templates**
| Template | Use Case | Description |
|----------|----------|-------------|
| `basic` | Quick prototyping | Fast recursive chunking |
| `advanced` | Production quality | Semantic-aware with quality scoring |
| `document_aware` | Structured documents | Respects sections and boundaries |
| `code_aware` | Source code | Function and class preservation |

### **Embedding Templates**
| Template | Model | Speed | Quality | Best For |
|----------|-------|-------|---------|----------|
| `sshuggingface_fast` | all-MiniLM-L6-v2 | ⚡⚡⚡ | ⭐⭐ | Prototyping |
| `huggingface_quality` | all-mpnet-base-v2 | ⚡⚡ | ⭐⭐⭐ | Production |
| `openai` | text-embedding-3-small | ⚡ | ⭐⭐⭐⭐ | High quality |
| `openai_large` | text-embedding-3-large | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

### **Vector Store Templates**
| Template | Index | Speed | Memory | Capacity |
|----------|-------|-------|---------|----------|
| `faiss_flat` | Flat | ⚡⚡⚡ | 📦📦 | 100K vectors |
| `faiss_hnsw` | HNSW | ⚡⚡ | 📦📦📦 | 1M vectors |
| `faiss_ivf` | IVF | ⚡ | 📦📦📦📦 | 100M+ vectors |
| `qdrant` | Qdrant | ⚡⚡ | 📦📦📦📦📦 | Production scale |

## 🛠️ Installation

### **Quick Install**
```bash
pip install neurosync
```

### **Development Install**
```bash
git clone https://github.com/nrnavaneet/neurosync.git
cd neurosync
pip install -e .
```

### **Docker Install**
```bash
docker pull neurosync/neurosync:latest
docker run -it neurosync/neurosync neurosync run /data
```

## Usage Examples

### **Interactive Mode (Recommended)**
```bash
neurosync run /path/to/documents
```
- Guided template selection for each phase
- API key collection and validation
- Real-time progress and metrics
- Interactive chat interface

### **Automatic Mode (Fast)**
```bash
neurosync run /path/to/documents --auto
```
- Smart defaults for all components
- Minimal user interaction
- Optimized for CI/CD pipelines

### **Custom Output Directory**
```bash
neurosync run /data --output-dir ./my-rag-project
```
- Organized project structure
- Reusable configurations
- Version control friendly

## 🔧 Advanced Configuration

### **Individual Phase Commands**
For fine-grained control, use individual commands:

```bash
# Phase 1: Ingestion
neurosync ingest file /docs --output ingested.json

# Phase 2: Processing
neurosync process file ingested.json --strategy advanced --output processed.json

# Phase 3: Embeddings & Vector Store
neurosync vector-store build processed.json embedding_config.json vector_config.json

# Phase 4: Search & Query
neurosync vector-store search "your query" embedding_config.json vector_config.json

# Phase 5: Serve API
neurosync serve api embedding_config.json vector_config.json
```

### **Configuration Files**
```bash
# Generate configuration templates
neurosync vector-store create-config embedding --model-type openai
neurosync vector-store create-config vector-store --store-type qdrant

# Use custom configurations
neurosync run /data --config custom-pipeline.yaml
```

## 🎯 Use Cases

### **Knowledge Management**
Build intelligent knowledge bases from:
- Technical documentation
- Company wikis and internal docs
- Research papers and publications
- Training materials and guides

### **Developer Tools**
Create code-aware applications:
- Code documentation search
- API reference chatbots
- Codebase Q&A systems
- Technical support assistants

### **Enterprise Applications**
Power business intelligence:
- Customer support knowledge bases
- Product documentation systems
- Internal training assistants
- Compliance and policy Q&A

### **Research & Analysis**
Accelerate research workflows:
- Literature review assistants
- Data analysis companions
- Hypothesis generation tools
- Citation and reference systems

---

**Built with by the NeuroSync team**

*Transform any data into intelligent, conversational AI applications with NeuroSync.*
