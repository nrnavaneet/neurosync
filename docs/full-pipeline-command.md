# NeuroSync Full Pipeline Command

The `neurosync run` command is the most powerful feature of NeuroSync - it provides a complete end-to-end pipeline from data ingestion to interactive chat, all with a single command.

## Quick Start

```bash
# Process a directory of documents
neurosync run /path/to/your/documents

# Process with automatic mode (no prompts)
neurosync run /path/to/your/documents --auto

# Process API data
neurosync run https://api.example.com/data

# Process database
neurosync run postgresql://user:pass@host:5432/dbname
```

## What It Does

The `neurosync run` command orchestrates all five phases of the NeuroSync pipeline:

### Phase 1: 🔄 Data Ingestion
- **Auto-detection**: Automatically detects whether your input is files, a directory, API endpoint, or database
- **Multiple sources**: Supports local files, directories, REST APIs, databases (PostgreSQL, MySQL, SQLite)
- **Smart templates**: Chooses optimal ingestion strategy based on your data type
- **Batch processing**: Efficiently processes large datasets
- **Error handling**: Robust error handling with detailed reporting

**Supported Input Types:**
- **Files**: `neurosync run document.pdf`
- **Directories**: `neurosync run /data/documents/`
- **APIs**: `neurosync run https://api.example.com/data`
- **Databases**: `neurosync run postgresql://user:pass@host:5432/db`

### Phase 2: 🧠 Intelligent Processing
- **Smart chunking**: Multiple strategies including recursive, semantic, and document-aware
- **Content analysis**: Automatic language detection and content classification
- **Quality scoring**: AI-powered assessment of chunk quality
- **Preprocessing**: Text cleaning, deduplication, and optimization

**Available Templates:**
- **Basic**: Simple recursive chunking for quick processing
- **Advanced**: Semantic-aware chunking with quality scoring
- **Document-Aware**: Respects document structure and boundaries
- **Code-Aware**: Specialized processing for code files

### Phase 3: ⚡ Embedding Generation
- **Multiple providers**: OpenAI, HuggingFace, Cohere, and more
- **Model selection**: Choose between speed and quality
- **Batch optimization**: Efficient batch processing
- **Monitoring**: Real-time performance metrics

**Available Templates:**
- **HuggingFace Fast**: Lightweight embeddings for quick processing
- **HuggingFace Quality**: High-quality embeddings for better accuracy
- **OpenAI**: State-of-the-art embeddings from OpenAI
- **OpenAI Large**: Maximum quality embeddings for critical applications

### Phase 4: 🗄️ Vector Store Creation
- **Auto-optimization**: Automatically selects optimal index type
- **Multiple backends**: FAISS (multiple index types) and Qdrant support
- **Scalability**: Handles datasets from thousands to millions of vectors
- **Versioning**: Built-in backup and version control

**Available Templates:**
- **FAISS Flat**: Simple, fast exact search for smaller datasets
- **FAISS HNSW**: High-performance approximate search for medium datasets
- **FAISS IVF**: Optimized for very large datasets
- **Qdrant**: Production-ready vector database

### Phase 5: 🤖 LLM Integration & Chat
- **Multi-provider**: OpenAI, Anthropic, Cohere, Google, and more
- **Fallback systems**: Automatic provider switching on failures
- **Context management**: Intelligent context retrieval
- **Interactive chat**: Real-time conversation interface

**Available Templates:**
- **OpenAI**: GPT-3.5 and GPT-4 models
- **Anthropic**: Claude models for safety-focused applications
- **Multi-Provider**: Robust setup with automatic fallbacks

## Template System

The pipeline uses intelligent templates that are automatically selected based on your input and requirements. You can also manually choose templates for each phase.

### Ingestion Templates

| Template            | Best For               | Description |
|-------------------- |------------------------|-------------|
| `file_basic`        | Small file collections | Basic file processing |
| `file_advanced`     | Large file collections | Enhanced processing with filters |
| `api_basic`         | REST APIs              | Standard API ingestion |
| `database_postgres` | PostgreSQL             | Optimized for PostgreSQL |
| `database_mysql`    | MySQL                  | Optimized for MySQL |
| `multi_source`      | Complex setups         | Combine multiple sources |

### Processing Templates

| Template | Best For | Description |
|----------|----------|-------------|
| `basic` | Quick processing | Simple recursive chunking |
| `advanced` | Quality focus | Semantic chunking with scoring |
| `document_aware` | Structured docs | Respects document boundaries |
| `code_aware` | Source code | Specialized for code files |

### Embedding Templates

| Template | Model | Speed | Quality | Best For |
|----------|-------|-------|---------|----------|
| `huggingface_fast` | all-MiniLM-L6-v2 | ⚡⚡⚡ | ⭐⭐ | Prototyping |
| `huggingface_quality` | all-mpnet-base-v2 | ⚡⚡ | ⭐⭐⭐ | Production |
| `openai` | text-embedding-3-small | ⚡ | ⭐⭐⭐⭐ | High quality |
| `openai_large` | text-embedding-3-large | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

### Vector Store Templates

| Template | Index Type | Speed | Memory | Best For |
|----------|------------|-------|---------|----------|
| `faiss_flat` | Flat | ⚡⚡⚡ | 📦📦 | < 100K vectors |
| `faiss_hnsw` | HNSW | ⚡⚡ | 📦📦📦 | < 1M vectors |
| `faiss_ivf` | IVF | ⚡ | 📦 | > 1M vectors |
| `qdrant` | Qdrant | ⚡⚡ | 📦📦 | Production use |

### LLM Templates

| Template | Providers | Best For |
|----------|-----------|----------|
| `openai` | OpenAI | General use, high quality |
| `anthropic` | Anthropic | Safety-focused applications |
| `multi_provider` | Multiple | Production with fallbacks |

## Command Options

### Basic Usage
```bash
neurosync run INPUT_PATH [OPTIONS]
```

### Options

- `--auto`: Run in automatic mode with smart defaults (no interactive prompts)
- `--output-dir`, `-o`: Specify output directory for generated files (default: current directory)

### Examples

#### Interactive Mode (Default)
```bash
neurosync run /path/to/documents
```
- Walks you through template selection for each phase
- Collects necessary API keys
- Shows progress and results for each phase
- Offers to start interactive chat

#### Automatic Mode
```bash
neurosync run /path/to/documents --auto
```
- Uses intelligent defaults for all templates
- Minimal user interaction (only API keys if needed)
- Fast processing with good defaults

#### Custom Output Directory
```bash
neurosync run /path/to/documents --output-dir ./my-project
```
- Creates all files in the specified directory
- Useful for organizing multiple projects

## API Key Management

The pipeline automatically detects which API keys you need based on your template choices:

### OpenAI (for embeddings or LLM)
```bash
# You'll be prompted to enter your OpenAI API key
# Get one from: https://platform.openai.com/api-keys
```

### Anthropic (for Claude models)
```bash
# You'll be prompted to enter your Anthropic API key
# Get one from: https://console.anthropic.com/
```

### Other Providers
The system will prompt for additional API keys as needed based on your selections.

## Output Files

The pipeline generates several configuration files that you can reuse:

- `config/embedding_config.json` - Embedding model configuration
- `config/vector_store_config.json` - Vector store settings
- `config/llm_config.json` - LLM provider configuration
- `data/phase1_ingested_*.json` - Raw ingested data
- `data/phase2_processed_*.json` - Processed and chunked data

## Performance Tips

### For Large Datasets
```bash
# Use automatic mode for faster processing
neurosync run /large/dataset --auto --output-dir ./large-project

# The pipeline will automatically:
# - Choose IVF indexing for large vector stores
# - Use efficient batch sizes
# - Select appropriate chunking strategies
```

### For High Quality
```bash
# Choose quality templates interactively
neurosync run /important/documents

# Select:
# - Advanced processing template
# - OpenAI large embeddings
# - HNSW or Qdrant vector store
# - Multi-provider LLM setup
```

### For Speed
```bash
# Use fast templates with auto mode
neurosync run /documents --auto

# Pipeline automatically selects:
# - Basic processing
# - Fast HuggingFace embeddings
# - Flat FAISS index (for small datasets)
# - Single LLM provider
```

## Monitoring and Debugging

The pipeline provides detailed progress information:

- **Phase Progress**: Real-time progress bars for each phase
- **Metrics**: Performance metrics and timing information
- **Error Handling**: Detailed error messages with suggestions
- **Results Summary**: Comprehensive summary of pipeline execution

## Integration with Other Commands

The generated configuration files can be used with individual NeuroSync commands:

```bash
# Use generated configs with individual commands
neurosync vector-store search "your query" \
  config/embedding_config.json \
  config/vector_store_config.json

# Start API server with generated configs
neurosync serve api config/embedding_config.json config/vector_store_config.json

# Add more data to existing vector store
neurosync vector-store build new_data.json \
  config/embedding_config.json \
  config/vector_store_config.json
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Verify API keys are correct and have sufficient credits
3. **Memory Issues**: Use smaller batch sizes or IVF indexing for large datasets
4. **Permission Issues**: Ensure write permissions in the output directory

### Getting Help

```bash
# Show command help
neurosync run --help

# Show general help
neurosync --help

# Enable verbose logging
neurosync --verbose run /path/to/data
```

The `neurosync run` command represents the culmination of NeuroSync's capabilities - providing a seamless, intelligent, and powerful way to build RAG applications from any data source with minimal effort.
