import fitz
import logging
import re
import io
import pytesseract
from PIL import Image
from typing import List
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OCR_LINE_GROUP_HEIGHT = 10
MIN_LINE_TEXT_LENGTH = 2

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
        self.setup_ocr()
    
    def _default_config(self) -> Dict:
        return {
            'min_text_length': 3,
            'ocr_confidence_threshold': 30,
            'heading_font_threshold': 14,
            'tesseract_config': '--oem 3 --psm 6',
            'ocr_zoom_factor': 2.0
        }
    
    def setup_ocr(self):
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except Exception as e:
            logger.warning(f"OCR not available: {e}")
            self.ocr_available = False
        
    def parse_pdf(self, file_path: str) -> List[ContentBlock]:
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Invalid file path: {file_path}")
        
        doc = fitz.open(str(file_path))
        content_blocks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            has_text = page.get_text("text").strip()
            has_images = bool(page.get_images(full=True))

            if not has_text and not has_images:
                logger.info(f"Page {page_num + 1}: Skipping (no text, no images)")
                continue

            if has_text:
                logger.info(f"Page {page_num + 1}: Extracting text blocks")
                text_blocks = self._process_text_blocks(page, page_num + 1)
                content_blocks.extend(text_blocks)
            
            if has_images:
                logger.info(f"Page {page_num + 1}: Extracting images")
                ocr_blocks = self._ocr_image(page, page_num + 1)
                content_blocks.extend(ocr_blocks)

        doc.close()
        return content_blocks
    
    def _process_text_blocks(self, page, page_num: int) -> List[ContentBlock]:
        if not page.get_text("text").strip():
            return []
        
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
    
    def _ocr_image(self, page, page_num: int) -> List[ContentBlock]:
        if not self.ocr_available:
            return []
        
        ocr_blocks = []
        
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.mode not in ['RGB', 'L']:
                        image = image.convert('RGB')
                    
                    image = image.convert('L')

                ocr_data = pytesseract.image_to_data(
                    image,
                    config=self.config['tesseract_config'],
                    output_type=pytesseract.Output.DICT
                )

                ocr_blocks.extend(self._group_ocr_text(ocr_data, page_num))
            
            except Exception as e:
                logger.error(f"OCR failed for image on page {page_num}: {e}")
        
        return ocr_blocks
    
    def _group_ocr_text(self, ocr_data: Dict, page_num: int) -> List[ContentBlock]:
        blocks = []
        lines = {}

        # Group text by line position
        for i, text in enumerate(ocr_data['text']):
            if not text.strip():
                continue
                
            confidence = int(ocr_data['conf'][i])
            # More lenient confidence threshold
            if confidence < 15:
                continue

            y = ocr_data['top'][i]
            line_key = round(y / OCR_LINE_GROUP_HEIGHT) * OCR_LINE_GROUP_HEIGHT
            lines.setdefault(line_key, []).append({
                'text': text,
                'confidence': confidence,
                'x': ocr_data['left'][i],
                'height': ocr_data['height'][i],
                'width': ocr_data['width'][i]
            })
        
        sorted_lines = sorted(lines.items())
        current_block = []
        last_line_y = None

        for line_y, line_items in sorted_lines:
            line_items.sort(key=lambda x: x['x'])
            line_text = ' '.join(item['text'] for item in line_items)

            if len(line_text.strip()) < MIN_LINE_TEXT_LENGTH:
                continue

            avg_conf = sum(item['confidence'] for item in line_items) / len(line_items)
            avg_height = sum(item['height'] for item in line_items) / len(line_items)
            
            # Check if this should start a new block
            should_start_new = self._should_start_new_block_improved(
                line_text, current_block, line_y, last_line_y, avg_height
            )

            if should_start_new:
                if current_block:
                    self._create_ocr_block(current_block, page_num, blocks)
                current_block = [(line_text, avg_conf)]
            else:
                current_block.append((line_text, avg_conf))
            
            last_line_y = line_y

        if current_block:
            self._create_ocr_block(current_block, page_num, blocks)
        
        return blocks
    
    def _should_start_new_block_improved(self, line_text: str, current_block: List, 
                                   line_y: int, last_line_y: int, avg_height: float) -> bool:
        if not current_block:
            return True
        
        # Large vertical gap suggests new section
        if last_line_y is not None and (line_y - last_line_y) > OCR_LINE_GROUP_HEIGHT * 2:
            return True
        
        # Check for headings (short, uppercase/title, or significantly larger font)
        words = line_text.split()
        if len(words) <= MAX_WORDS_FOR_HEADING:
            if (line_text.isupper() or line_text.istitle()) and len(words) > 1:
                # Avoid false positives for common paragraph starters
                if not any(line_text.lower().startswith(word) for word in 
                        ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'they', 'this', 'that']):
                    return True
        
        # List items
        if LIST_ITEM_REGEX.match(line_text):
            return True
        
        # Bullet points without regex (common OCR misreads)
        if line_text.strip().startswith(('•', '·', '-', '*')) and len(words) > 1:
            return True
        
        # Continue current block for regular text
        return False
    
    def _create_ocr_block(self, block_lines: List, page_num: int, blocks: List):
        block_text = '\n'.join(line[0] for line in block_lines)
        avg_conf = sum(line[1] for line in block_lines) / len(block_lines)

        if len(block_text.strip()) >= self.config['min_text_length']:
            content_type = self._classify_content_type(block_text)
            blocks.append(ContentBlock(
                content=block_text.strip(),
                content_type=content_type,
                page_number=page_num,
                confidence=avg_conf / 100.0,
                metadata={'source': 'ocr', 'line_count': len(block_lines)}
            ))
    
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

