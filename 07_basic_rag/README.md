# Basic Local RAG

This project implements a local retrieval-based question-answering pipeline using synthetic model governance content.

## Architecture

Question → TF-IDF Vector → Similarity Search → Top-K Chunks → Grounded Answer

## Key Features

- Synthetic knowledge base
- Paragraph-based chunking
- Local vector store
- Cosine-similarity retrieval
- Top-k document selection
- Source-based responses
- Fallback for unsupported questions

## Limitation

The current version does not use an external generative model. It returns retrieved source content as a grounded answer. This design allows the retrieval and grounding logic to be tested independently.