# NeuroSync Architecture Documentation

## Overview
NeuroSync is an AI-native ETL pipeline designed for RAG and LLM applications. It provides intelligent data ingestion, processing, and chunking capabilities.

## Core Components

### 1. Ingestion Layer
- **File Connector**: Processes local files (PDF, TXT, MD, etc.)
- **API Connector**: Fetches data from REST APIs
- **Database Connector**: Connects to SQL/NoSQL databases

### 2. Processing Layer
- **Recursive Chunker**: Text-based recursive splitting
- **Semantic Chunker**: Sentence-aware chunking using spaCy
- **Sliding Window Chunker**: Fixed-size overlapping chunks
- **Token-Aware Chunker**: Token-boundary aware splitting
- **Hierarchical Chunker**: Document structure-aware processing
- **Document Structure Chunker**: Advanced structure analysis

### 3. Storage Layer
- **Vector Store**: High-performance vector database
- **Metadata Store**: Rich metadata preservation
- **Index Management**: Optimized search and retrieval

## Pipeline Strategies

### Recursive Strategy
Best for general-purpose text processing with balanced chunk sizes.

### Semantic Strategy
Ideal for maintaining semantic coherence in chunks.

### Sliding Window Strategy
Ensures no information loss with overlapping chunks.

### Token-Aware Strategy
Optimized for token-based models with precise boundaries.

### Hierarchical Strategy
Preserves document hierarchy and relationships.

### Document Structure Strategy
Advanced structure analysis for complex documents.

## Performance Characteristics

| Strategy | Chunk Count | Processing Speed | Use Case |
|----------|-------------|------------------|----------|
| Recursive | Medium | Fast | General purpose |
| Semantic | Low | Medium | Coherence-focused |
| Sliding | High | Fast | No information loss |
| Token-Aware | High | Medium | Token optimization |
| Hierarchical | High | Medium | Structure preservation |
| Document Structure | High | Slow | Complex documents |

## Configuration

NeuroSync uses JSON-based configuration files for pipeline definition:

```json
{
  "name": "example_pipeline",
  "ingestion": {
    "sources": [
      {
        "name": "docs",
        "path": "docs/",
        "type": "file"
      }
    ]
  },
  "processing": {
    "strategy": "semantic",
    "chunk_size": 1024,
    "chunk_overlap": 200
  }
}
```

## Integration

NeuroSync integrates with:
- Vector databases (Chroma, Pinecone, Weaviate)
- ML frameworks (spaCy, transformers)
- Cloud services (AWS, GCP, Azure)
- Monitoring tools (Airflow, MLflow)
