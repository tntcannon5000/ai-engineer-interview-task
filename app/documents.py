"""Bounded local text extraction; uploaded files are never persisted."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from pypdf import PdfReader

MAX_BYTES = 10 * 1024 * 1024
MAX_CHARS = 100_000
EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in EXTENSIONS:
        raise ValueError("Supported formats: TXT, Markdown (.md), PDF and DOCX")
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose a non-empty file up to 10 MB")
    try:
        if extension in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
            if any(ord(c) < 32 and c not in "\n\r\t" for c in text):
                raise ValueError("Text files must contain readable UTF-8 text")
        elif extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ValueError("File is not a PDF")
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported")
            if len(reader.pages) > 100:
                raise ValueError("PDFs must have at most 100 pages")
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
                if sum(map(len, parts)) > MAX_CHARS:
                    raise ValueError("Extracted text exceeds 100,000 characters")
            text = "\n\n".join(parts)
        else:
            with ZipFile(BytesIO(content)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 40 * 1024 * 1024:
                    raise ValueError("DOCX expands beyond the 40 MB limit")
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("File is not a DOCX document")
            document = Document(BytesIO(content))
            parts = []
            for block in document.iter_inner_content():
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
            text = "\n".join(parts)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Unable to read this document; check its format or export it as text") from exc
    if not text.strip():
        raise ValueError("No readable text found. Scanned documents need OCR; paste their text instead.")
    if len(text) > MAX_CHARS:
        raise ValueError("Extracted text exceeds 100,000 characters")
    return text
