import numpy as np


def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors."""
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


model_validation = np.array([0.9, 0.8, 0.2])
model_risk = np.array([0.8, 0.9, 0.3])
customer_service = np.array([0.1, 0.2, 0.9])

comparisons = {
    "Model Risk": model_risk,
    "Customer Service": customer_service,
}

print("=== Cosine Similarity Results ===")

for name, vector in comparisons.items():
    score = cosine_similarity(model_validation, vector)
    print(f"Model Validation vs {name}: {score:.4f}")