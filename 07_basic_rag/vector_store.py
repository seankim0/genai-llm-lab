
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,       # Convert all text to lowercase
    stop_words="english",  # Remove common English stop words
)

# Convert text chunks into TF-IDF feature vectors
chunk_vectors = vectorizer.fit_transform(chunks)

# Display vector store statistics
print("=== Local Vector Store ===")
print(f"Chunks: {len(chunks)}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Vector shape: {chunk_vectors.shape}")
