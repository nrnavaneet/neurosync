# Sample API Data for Testing

## Overview
This document contains sample API response data for testing NeuroSync's API ingestion capabilities.

## REST API Endpoints

### GET /api/v1/documents
Returns a list of documents with metadata:

```json
{
  "status": "success",
  "total": 150,
  "page": 1,
  "per_page": 10,
  "data": [
    {
      "id": "doc_001",
      "title": "Introduction to Machine Learning",
      "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience...",
      "author": "Dr. Sarah Johnson",
      "created_at": "2024-01-15T10:30:00Z",
      "tags": ["machine-learning", "ai", "algorithms"],
      "category": "education",
      "word_count": 2500
    },
    {
      "id": "doc_002",
      "title": "Deep Learning Fundamentals",
      "content": "Deep learning is a machine learning technique that teaches computers to learn by example. In deep learning, a computer model learns to perform classification tasks directly from images, text, or sound...",
      "author": "Prof. Michael Chen",
      "created_at": "2024-01-16T14:22:00Z",
      "tags": ["deep-learning", "neural-networks", "tensorflow"],
      "category": "research",
      "word_count": 3200
    }
  ]
}
```

### GET /api/v1/articles
Returns article content:

```json
{
  "articles": [
    {
      "id": "art_001",
      "headline": "The Future of Artificial Intelligence in Healthcare",
      "body": "Artificial intelligence is revolutionizing healthcare by enabling more accurate diagnoses, personalized treatment plans, and efficient drug discovery. Recent advances in machine learning have shown remarkable success in medical imaging, where AI systems can detect diseases like cancer with accuracy matching or exceeding human specialists. The integration of AI in electronic health records is streamlining clinical workflows and reducing administrative burden on healthcare providers. Natural language processing is helping extract insights from unstructured medical notes, while predictive analytics are identifying patients at risk of complications. However, challenges remain in ensuring AI fairness, maintaining patient privacy, and achieving regulatory approval for AI-powered medical devices.",
      "published_date": "2024-02-01",
      "author": "Dr. Emily Rodriguez",
      "source": "Medical AI Quarterly",
      "topics": ["healthcare", "artificial-intelligence", "medical-imaging"],
      "reading_time": 8
    },
    {
      "id": "art_002",
      "headline": "Sustainable Technology Solutions for Climate Change",
      "body": "Climate change represents one of the most pressing challenges of our time, requiring innovative technological solutions to reduce greenhouse gas emissions and build resilience against environmental impacts. Renewable energy technologies like solar panels and wind turbines have achieved unprecedented efficiency and cost-effectiveness, making them competitive with fossil fuels in many markets. Smart grid systems are optimizing energy distribution and enabling integration of distributed renewable sources. Carbon capture and storage technologies are being developed to remove CO2 from the atmosphere and industrial processes. Advanced materials science is creating more efficient batteries for energy storage and electric vehicles. Artificial intelligence is optimizing energy consumption in buildings and transportation systems. However, scaling these solutions globally requires significant investment, policy support, and international cooperation.",
      "published_date": "2024-02-03",
      "author": "Dr. James Liu",
      "source": "Green Tech Review",
      "topics": ["climate-change", "renewable-energy", "sustainability"],
      "reading_time": 10
    }
  ]
}
```

### GET /api/v1/research
Returns research paper abstracts:

```json
{
  "papers": [
    {
      "arxiv_id": "2024.0156",
      "title": "Attention Mechanisms in Large Language Models: A Comprehensive Survey",
      "abstract": "Large Language Models (LLMs) have achieved remarkable success across various natural language processing tasks, largely due to the effectiveness of attention mechanisms. This survey provides a comprehensive overview of attention mechanisms used in modern LLMs, including self-attention, multi-head attention, and sparse attention patterns. We analyze the computational complexity, memory requirements, and performance characteristics of different attention variants. Our review covers recent innovations such as sliding window attention, local attention, and attention with learned patterns. We also discuss the trade-offs between attention complexity and model performance, providing insights for practitioners designing efficient LLM architectures. The survey includes empirical comparisons across different attention mechanisms on standard benchmarks, highlighting the strengths and limitations of each approach. We conclude with a discussion of future research directions in attention mechanism design for next-generation language models.",
      "authors": ["Alice Wang", "Bob Zhang", "Carol Li"],
      "published": "2024-01-20",
      "categories": ["cs.CL", "cs.AI", "cs.LG"],
      "journal": "Conference on Neural Information Processing Systems",
      "citation_count": 42
    },
    {
      "arxiv_id": "2024.0234",
      "title": "Efficient Retrieval-Augmented Generation for Long-Context Applications",
      "abstract": "Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing language models with external knowledge. However, existing RAG systems face challenges when dealing with long-context applications requiring reasoning over extensive document collections. This paper introduces novel techniques for efficient RAG in long-context scenarios, including hierarchical retrieval strategies, adaptive chunking methods, and attention-aware relevance scoring. Our approach reduces computational overhead by 40% while maintaining or improving generation quality on long-context benchmarks. We propose a multi-stage retrieval pipeline that first identifies relevant document sections, then performs fine-grained passage retrieval within selected sections. The hierarchical approach enables processing of document collections exceeding 100,000 pages while maintaining sub-second response times. Experimental results on question-answering, summarization, and dialogue tasks demonstrate the effectiveness of our methods across diverse long-context applications. We also provide analysis of the trade-offs between retrieval depth, computational cost, and generation quality.",
      "authors": ["David Kumar", "Eva Petrov", "Frank Mueller"],
      "published": "2024-01-25",
      "categories": ["cs.CL", "cs.IR", "cs.AI"],
      "journal": "International Conference on Machine Learning",
      "citation_count": 28
    }
  ]
}
```

## Sample Database Schema

### Documents Table
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    author VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category VARCHAR(50),
    tags TEXT[],
    word_count INTEGER,
    language VARCHAR(10) DEFAULT 'en',
    status VARCHAR(20) DEFAULT 'published'
);
```

### Sample Data
```sql
INSERT INTO documents (title, content, author, category, tags, word_count) VALUES
('Natural Language Processing Fundamentals', 'Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a manner that is valuable. NLP combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to process and analyze large amounts of natural language data...', 'Dr. Lisa Anderson', 'education', ARRAY['nlp', 'ai', 'linguistics'], 1800),

('Computer Vision Applications in Autonomous Vehicles', 'Computer vision plays a crucial role in the development of autonomous vehicles, enabling them to perceive and interpret their environment. Advanced image processing algorithms, combined with deep learning techniques, allow self-driving cars to detect objects, recognize traffic signs, track pedestrians, and navigate complex road scenarios. The integration of multiple sensors including cameras, LiDAR, and radar creates a comprehensive perception system that ensures safe autonomous navigation...', 'Prof. Robert Kim', 'research', ARRAY['computer-vision', 'autonomous-vehicles', 'deep-learning'], 2200),

('Blockchain Technology and Decentralized Applications', 'Blockchain technology has revolutionized the way we think about data storage, security, and decentralized systems. Originally developed as the underlying technology for Bitcoin, blockchain has found applications across various industries including finance, supply chain, healthcare, and digital identity management. The immutable and transparent nature of blockchain makes it ideal for creating trustless systems where parties can interact without requiring a central authority...', 'Maria Garcia', 'technology', ARRAY['blockchain', 'cryptocurrency', 'decentralization'], 1950);
```

This sample data provides realistic content for testing all ingestion methods and processing strategies.
