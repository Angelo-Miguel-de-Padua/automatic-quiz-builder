import fitz
import logging
import re
from typing import List
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

LIST_ITEM_REGEX = re.compile(r'^\s*[\d\-\•\*]')

BOLD_FONT_FLAG = 1 << 4
MAX_WORDS_FOR_HEADING = 10

class ContentType(str, Enum):
    HEADING = 'heading'
    PARAGRAPH = 'paragraph'
    LIST = 'list'
    TABLE = 'table'
    FORMULA = 'formula'
    CODE = 'code'
    IMAGE_CAPTION = 'image_caption'
    DEFINITION = 'definition'
    THEOREM = 'theorem'
    EXAMPLE = 'example'


@dataclass
class ContentBlock:
    content: str
    content_type: str
    page_number: int
    confidence: float
    metadata: Dict[str, Any] = None
    original_content: str = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.original_content is None:
            self.original_content = self.content

class PDFParser:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            'heading_font_threshold': 14
        }
        
    def parse_pdf(self, file_path: str) -> List[ContentBlock]:
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Invalid file path: {file_path}")
        
        doc = fitz.open(str(file_path))
        content_blocks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = self._process_text_blocks(page, page_num + 1)
            content_blocks.extend(blocks)

        doc.close()
        return content_blocks
    
    def _process_text_blocks(self, page, page_num: int) -> List[ContentBlock]:
        blocks = []
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            
            block_text = ""
            font_sizes = []
            font_flags = []

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        block_text += text + " "
                        font_sizes.append(span.get("size", 12))
                        font_flags.append(span.get("flags", 0))
            
            cleaned_text = self.clean_text(block_text)
            if cleaned_text:
                content_type = self._classify_content_type(cleaned_text, font_sizes, font_flags)
                blocks.append(ContentBlock(
                    content=cleaned_text,
                    content_type=content_type,
                    page_number=page_num,
                    confidence=1.0,
                    metadata={'font_sizes': font_sizes, 'font_flags': font_flags}
                ))
        
        return blocks
    
    def _classify_content_type(self, text: str, font_sizes=None, font_flags=None) -> str:
        text_lower = text.lower().strip()

        if font_sizes and sum(font_sizes) / len(font_sizes) > self.config.get('heading_font_threshold', 14):
            return ContentType.HEADING
        
        if font_flags and any(flag & BOLD_FONT_FLAG for flag in font_flags):
            if len(text.split()) < MAX_WORDS_FOR_HEADING:
                return ContentType.HEADING
        
        if len(text.split()) < MAX_WORDS_FOR_HEADING and (text.isupper() or text.istitle()):
            return ContentType.HEADING
        
        if LIST_ITEM_REGEX.match(text):
            return ContentType.LIST
        
        if any(sym in text for sym in ['∑', '∫', '=', '≠', '√']):
            return ContentType.FORMULA
        
        if 'def ' in text_lower or 'class ' in text_lower or 'return ' in text_lower:
            return ContentType.CODE
        
        if any(phrase in text_lower for phrase in ['is defined as', 'refers to', 'means that']):
            return ContentType.DEFINITION
        
        return ContentType.PARAGRAPH
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)

        return text.strip()
        
