
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LocalRetriever:
    """Retrieve relevant text chunks using TF-IDF similarity."""

    def __init__(self, documents):
        # Store the original document chunks
        self.documents = documents

        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        # Convert document chunks into TF-IDF vectors
        self.document_vectors = self.vectorizer.fit_transform(
            documents
        )

    def search(self, query, top_k=3):
        # Convert the search query into a TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity between query and documents
        scores = cosine_similarity(
            query_vector,
            self.document_vectors,
        ).flatten()

        # Rank documents by similarity score (highest first)
        ranked_indices = scores.argsort()[::-1][:top_k]

        # Return the top-K documents with similarity scores
        return [
            {
                "document": self.documents[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]


# Define the path to the knowledge base file
file_path = Path(__file__).parent / "knowledge_base.txt"

# Read the knowledge base using UTF-8 encoding
text = file_path.read_text(encoding="utf-8")

# Split the document into non-empty paragraph chunks
chunks = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

# Initialize the local TF-IDF retriever
retriever = LocalRetriever(chunks)

# Define the search query
query = "What should model validation evaluate?"

# Retrieve the top 3 most relevant document chunks
results = retriever.search(query, top_k=3)

# Display the search query
print(f"Query: {query}\n")

# Display retrieved documents and their similarity scores
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(result["document"])
    print()
