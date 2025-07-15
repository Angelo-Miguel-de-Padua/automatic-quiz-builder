import fitz
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class ContextBlock:
    content: str
    content_type: str
    page_number: int
    confidence: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PDFParser:
    def parse_pdf(self, file_path: str) -> List[ContextBlock]:
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Invalid file path: {file_path}")
        
        doc = fitz.open(str(file_path))
        content_blocks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                block = ContextBlock(
                    content=text.strip(),
                    content_type="paragraph",
                    page_number=page_num + 1,
                    confidence=1.0,
                    metadata={"source": "raw_text"}
                )
                content_blocks.append(block)

        doc.close()
        return content_blocks
