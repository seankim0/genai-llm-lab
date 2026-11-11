from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the documents to search.
documents = [
    "Independent model validation evaluates model performance.",
    "Model governance defines roles and responsibilities.",
    "Hallucination occurs when an LLM generates unsupported facts.",
    "RAG retrieves documents before generating a response.",
    "Customer service chatbots respond to user questions.",
]

# Define the search query.
query = "How should an organization validate a model?"

# Configure the TF-IDF vectorizer.
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
)

# Convert the documents and query into TF-IDF vectors.
all_text = documents + [query]
vectors = vectorizer.fit_transform(all_text)

# Separate the document vectors from the query vector.
document_vectors = vectors[:-1]
query_vector = vectors[-1]

# Calculate cosine similarity between the query and each document.
scores = cosine_similarity(
    query_vector,
    document_vectors,
).flatten()

# Rank the documents by similarity score in descending order.
ranked_results = sorted(
    enumerate(scores),
    key=lambda item: item[1],
    reverse=True,
)

# Display the query and ranked search results.
print(f"Query: {query}\n")
print("=== Search Results ===")

for index, score in ranked_results:
    print(f"Score: {score:.4f}")
    print(f"Document: {documents[index]}\n")