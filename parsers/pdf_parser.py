import fitz
import logging
import io
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
from typing import List
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from utils.content_utils import ContentBlock, create_content_block, group_words_into_lines, group_lines_into_blocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OCR_LINE_GROUP_HEIGHT = 10
MIN_LINE_TEXT_LENGTH = 2
CONTRAST_ENHANCEMENT_FACTOR = 1.2
THRESHOLD_OFFSET = 5
PIXEL_WHITE = 255
PIXEL_BLACK = 0

class TextExtractor:
    """Handles direct text extraction from PDF text layers"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'min_text_length': 3,
            'heading_font_threshold': 14,
            'bold_font_flag': 1 << 4,
            'max_words_for_heading': 10
        }
    
    def clean_ocr_text(self, text):
        if not text.strip():
            return ""
        return " ".join(text)
    
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
            
            block = create_content_block(
                content=block_text,
                page_num=page_num,
                confidence=1.0,
                source='text',
                metadata={'font_sizes': font_sizes, 'font_flags': font_flags},
                config=self.config
            )

            if block:
                blocks.append(block)
        
        return blocks
    
class OCRProcessor:
    """Handles OCR extraction from images and image-based PDFs"""
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(lang=lang)
        self.text_extractor = TextExtractor()
    
    def process_image(self, img_array, page_num):
        try:
            result = self.ocr.predict(img_array)
            text_bbox_pairs = []
            combined_text = []

            if isinstance(result, list) and len(result) > 0:
                page_result = result[0]
                if isinstance(page_result, dict):
                    texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [1.0] * len(texts))
                    bboxes = page_result.get('det_polys', [])

                    for j, (text, score) in enumerate(zip(texts,)):
                        if text and text.strip():
                            cleaned_text = self.text_extractor.clean_ocr_text(text.strip())
                            combined_text.append(cleaned_text)
                            bbox = self._extract_bbox(bboxes, j)
                            text_bbox_pairs.append({
                                'text': cleaned_text,
                                'bbox': bbox,
                                'confidence': float(score)
                            })
                else:
                    print(f"Unexpected page result structure: {type(page_result)}")
            
            ocr_text = " ".join(combined_text) if combined_text else ""
            return {
                'page': page_num,
                'text': ocr_text,
                'source': 'ocr',
                'text_bbox_pairs': text_bbox_pairs
            }
        
        except Exception as e:
            print(f"OCR failed: {e}")
            return {
                'page': page_num,
                'text': "",
                'source': 'ocr',
                'text_bbox_pairs': []
            }
    
class PDFParser:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.text_extractor = TextExtractor(self.config.get('text_extraction', {}))
        self.ocr_extractor = OCRProcessor(self.config.get('ocr', {}))
        self.ocr_extractor.setup_ocr()

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
                text_blocks = self.text_extractor._process_text_blocks(page, page_num + 1)
                content_blocks.extend(text_blocks)
            
            if has_images:
                logger.info(f"Page {page_num + 1}: Extracting images")
                ocr_blocks = self.ocr_extractor.extract_ocr_blocks(page, page_num + 1)
                content_blocks.extend(ocr_blocks)

        doc.close()
        return content_blocks