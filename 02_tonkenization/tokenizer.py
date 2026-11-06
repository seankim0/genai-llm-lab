from transformers import AutoTokenizer

# Load a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample text
text = "Model validation is important for managing AI model risk."

# --------------------------------------------------
# Step 1: Simple Python tokenization
# --------------------------------------------------

simple_tokens = text.lower().split()
print("=== Original Text ===")
print(text)
print("\n=== Simple Python Tokens ===")
print(simple_tokens)

# --------------------------------------------------
# Step 2: Hugging Face tokenization
# --------------------------------------------------

llm_tokens = tokenizer.tokenize(text)
print("\n=== Hugging Face Tokens ===")
print(llm_tokens)

# --------------------------------------------------
# Step 3: Convert tokens to token IDs
# --------------------------------------------------

token_ids = tokenizer.encode(text)
print("\n=== Token IDs ===")
print(token_ids)

# --------------------------------------------------
# Step 4: Decode token IDs back to text
# --------------------------------------------------

decoded_text = tokenizer.decode(token_ids)
print("\n=== Decoded Text ===")
print(decoded_text)

# --------------------------------------------------
# Step 5: Compare different words
# --------------------------------------------------

test_words = [
"validation",
"governance",
"hallucination",
"tokenization",
"generative",
]

print("\n=== Word Tokenization Examples ===")

for word in test_words:
    tokens = tokenizer.tokenize(word)

print(f"{word}: {tokens}")