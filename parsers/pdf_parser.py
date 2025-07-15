import fitz
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFParser:
    def parse_pdf(self, file_path: str) -> List[str]:
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Invalid file path: {file_path}")
        
        doc = fitz.open(str(file_path))
        raw_pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            raw_pages.append(text)

        doc.close()
        return raw_pages
