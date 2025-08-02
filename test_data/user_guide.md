# NeuroSync User Guide

Welcome to NeuroSync - the AI-native ETL pipeline for modern RAG applications.

## Getting Started

### Installation
```bash
pip install neurosync
```

### Quick Start
1. Initialize a new project: `neurosync init my_project`
2. Create a pipeline configuration
3. Run ingestion: `neurosync ingest file data/`
4. Process data: `neurosync pipeline run config.json`

## Data Ingestion

### File Ingestion
Ingest local files and directories:
```bash
neurosync ingest file documents/ -r --output results/
```

Supported formats:
- Text files (.txt, .md)
- PDFs (.pdf)
- Office documents (.docx, .xlsx)
- Web formats (.html, .xml)
- Data formats (.json, .csv)

### API Ingestion
Fetch data from REST APIs:
```bash
neurosync ingest api https://api.example.com/data --auth-token TOKEN
```

Features:
- Authentication support (Bearer, API Key, OAuth)
- Rate limiting and retry logic
- Pagination handling
- Custom headers and parameters

### Database Ingestion
Connect to databases:
```bash
neurosync ingest database --connection-string "postgresql://user:pass@host:5432/db"
```

Supported databases:
- PostgreSQL
- MySQL
- SQLite
- MongoDB
- Redis

## Processing Strategies

### Choosing a Strategy

**Recursive**: General-purpose text splitting
- Use for: Mixed content, balanced performance
- Chunk size: 1024 tokens
- Overlap: 200 tokens

**Semantic**: Sentence-boundary aware
- Use for: Coherent meaning preservation
- Model: spaCy language models
- Best for: Long-form text, articles

**Sliding Window**: Fixed-size overlapping chunks
- Use for: Ensuring no information loss
- High recall applications
- Search-oriented processing

**Token-Aware**: Token-boundary optimization
- Use for: LLM-specific applications
- Precise token counting
- Model-specific optimization

**Hierarchical**: Document structure preservation
- Use for: Structured documents
- Maintains relationships
- Section-aware processing

**Document Structure**: Advanced structure analysis
- Use for: Complex documents
- Table and list handling
- Rich formatting preservation

## Pipeline Configuration

### Basic Configuration
```json
{
  "name": "basic_pipeline",
  "ingestion": {
    "sources": [
      {
        "name": "docs",
        "path": "documents/",
        "type": "file"
      }
    ]
  },
  "processing": {
    "strategy": "recursive",
    "chunk_size": 1024,
    "chunk_overlap": 200
  },
  "output": "results.json"
}
```

### Advanced Configuration
```json
{
  "name": "advanced_pipeline",
  "ingestion": {
    "sources": [
      {
        "name": "files",
        "path": "data/",
        "type": "file",
        "recursive": true,
        "filters": ["*.pdf", "*.txt"]
      },
      {
        "name": "api_data",
        "endpoint": "https://api.example.com/content",
        "type": "api",
        "auth": {
          "type": "bearer",
          "token": "${API_TOKEN}"
        }
      },
      {
        "name": "db_data",
        "connection": "postgresql://localhost:5432/content",
        "query": "SELECT * FROM documents",
        "type": "database"
      }
    ]
  },
  "processing": {
    "strategy": "hierarchical",
    "chunk_size": 768,
    "chunk_overlap": 150,
    "model": "en_core_web_sm",
    "preprocessing": [
      {"name": "html_cleaner", "enabled": true},
      {"name": "whitespace_normalizer", "enabled": true}
    ]
  },
  "output": {
    "format": "json",
    "path": "processed_data.json",
    "include_metadata": true,
    "chunk_validation": true
  }
}
```

## Best Practices

### Performance Optimization
1. **Choose appropriate chunk sizes**: 512-1024 tokens for most use cases
2. **Use preprocessing**: Clean HTML, normalize whitespace
3. **Batch processing**: Process multiple files together
4. **Monitor memory usage**: Large files may need streaming

### Quality Assurance
1. **Validate configurations**: Use `neurosync ingest validate-config`
2. **Test with small datasets**: Verify output before large runs
3. **Monitor chunk quality**: Check semantic coherence
4. **Review metadata**: Ensure proper source attribution

### Troubleshooting
1. **Check logs**: Use `--verbose` flag for detailed output
2. **Validate inputs**: Ensure files are readable and formatted correctly
3. **Test connections**: Use connection test commands
4. **Memory issues**: Reduce chunk sizes or use streaming mode

## Integration Examples

### With Vector Databases
```python
from neurosync import NeuroSyncPipeline
from chromadb import Client

pipeline = NeuroSyncPipeline("config.json")
results = pipeline.run()

client = Client()
collection = client.create_collection("documents")

for chunk in results.chunks:
    collection.add(
        documents=[chunk.content],
        metadatas=[chunk.metadata],
        ids=[chunk.chunk_id]
    )
```

### With LangChain
```python
from langchain.text_splitter import NeuroSyncSplitter
from langchain.vectorstores import Chroma

splitter = NeuroSyncSplitter(
    strategy="semantic",
    chunk_size=1024,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(docs, embeddings)
```

## Support and Resources

- Documentation: https://neurosync.readthedocs.io
- GitHub: https://github.com/neurosync/neurosync
- Community: https://discord.gg/neurosync
- Issues: https://github.com/neurosync/neurosync/issues
