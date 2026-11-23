
from pathlib import Path

# Define the path to the knowledge base file
file_path = Path(__file__).parent / "knowledge_base.txt"

# Read the file content using UTF-8 encoding
text = file_path.read_text(encoding="utf-8")


def chunk_by_paragraph(document):
    """Split a document into non-empty paragraphs."""

    # Split the document by blank lines and remove whitespace
    return [
        paragraph.strip()
        for paragraph in document.split("\n\n")
        if paragraph.strip()  # Exclude empty paragraphs
    ]


# Split the knowledge base into paragraph-based chunks
chunks = chunk_by_paragraph(text)

# Display the total number of chunks
print(f"Number of chunks: {len(chunks)}\n")

# Print each chunk with its index, starting from 1
for index, chunk in enumerate(chunks, start=1):
    print(f"Chunk {index}:")
    print(chunk)
    print()  # Add a blank line between chunks
