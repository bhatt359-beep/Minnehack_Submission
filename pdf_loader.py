import PyPDF2
import re

def get_pdf_text(pdf_path):
    """
    Extract text from a PDF file.
    Returns the full text as a single string.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip() if text else None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of approximately chunk_size characters,
    with optional overlap between chunks.
    Returns a list of strings.
    """
    if not text:
        return []

    # Clean the text
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with single space

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # Move start point with overlap

    return chunks