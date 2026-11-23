
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BasicRAG:
    """A local retrieval-based question answering pipeline."""

    def __init__(self, documents):
        # Store the original document chunks
        self.documents = documents

        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        # Convert documents into TF-IDF vectors
        self.document_vectors = self.vectorizer.fit_transform(
            documents
        )

    def retrieve(self, question, top_k=2):
        # Convert the question into a TF-IDF vector
        question_vector = self.vectorizer.transform([question])

        # Calculate cosine similarity between question and documents
        scores = cosine_similarity(
            question_vector,
            self.document_vectors,
        ).flatten()

        # Rank documents by similarity score (highest first)
        ranked_indices = scores.argsort()[::-1][:top_k]

        # Return relevant documents with positive similarity scores
        return [
            {
                "text": self.documents[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
            if scores[index] > 0
        ]

    def answer(self, question, top_k=2):
        # Retrieve the most relevant document chunks
        sources = self.retrieve(question, top_k)

        # Return a fallback response if no relevant sources are found
        if not sources:
            return {
                "answer": (
                    "I could not find sufficient supporting "
                    "information in the knowledge base."
                ),
                "sources": [],
            }

        # Combine retrieved documents into a single context
        context = " ".join(source["text"] for source in sources)

        # Return the retrieved context and supporting sources
        return {
            "answer": context,
            "sources": sources,
        }


# Define the path to the knowledge base file
file_path = Path(__file__).parent / "knowledge_base.txt"

# Read the knowledge base using UTF-8 encoding
text = file_path.read_text(encoding="utf-8")

# Split the document into non-empty paragraph chunks
documents = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

# Initialize the Basic RAG pipeline
rag = BasicRAG(documents)

# Define the user question
question = "What is LLM hallucination?"

# Retrieve relevant documents and generate the response
response = rag.answer(question)

# Display the question and retrieved answer
print(f"Question: {question}\n")

print("Answer:")
print(response["answer"])

# Display supporting sources and similarity scores
print("\nSources:")

for source in response["sources"]:
    print(f"- Score: {source['score']:.4f}")
    print(f"  {source['text']}")
