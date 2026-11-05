import re
from collections import Counter


# Sample text for NLP practice
# text = """
# Model validation is an important part of model risk management.
# Model validation helps identify model risk and supports effective model governance.
# """

text = """
Large language models introduce new model risks.
LLM validation should evaluate hallucination, accuracy, robustness,
and model governance requirements.
Model risk management provides an important framework
for validating generative AI models.
"""


# Step 1: Convert text to lowercase
clean_text = text.lower()

print("=== Original Text ===")
print(text)

print("=== Lowercase Text ===")
print(clean_text)


# Step 2: Remove punctuation
clean_text = re.sub(r"[^a-z0-9\s]", "", clean_text)

print("=== Cleaned Text ===")
print(clean_text)


# Step 3: Tokenize text into individual words
tokens = clean_text.split()

print("=== Tokens ===")
print(tokens)


# Step 4: Count word frequencies
word_counts = Counter(tokens)

print("=== Word Frequency ===")
for word, count in word_counts.most_common():
    print(f"{word}: {count}")