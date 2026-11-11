from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Model validation evaluates model performance and risk.",
    "Model risk management provides governance and oversight.",
    "Customer service chatbots answer customer questions.",
    "RAG retrieves relevant documents before generating answers.",
]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
)

document_vectors = vectorizer.fit_transform(documents)

print("=== Vocabulary ===")
print(vectorizer.get_feature_names_out())

print("\n=== Vector Shape ===")
print(document_vectors.shape)

print("\n=== TF-IDF Matrix ===")
print(document_vectors.toarray())