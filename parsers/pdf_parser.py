import fitz
import logging
import re
import io
import pytesseract
from PIL import Image, ImageEnhance
from typing import List
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import numpy as np

from utils.content_utils import ContentBlock, ContentType, classify_content, clean_text, create_content_block

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
    
class OCRExtractor:
    """Handles OCR extraction from images and image-based PDFs"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'min_text_length': 3,
            'ocr_confidence_threshold': 30,
            'tesseract_config': '--oem 3 --psm 6',
            'ocr_zoom_factor': 3.0,
            'enhance_contrast': True,
            'line_height_threshold': 10,
            'word_spacing_threshold': 20,
            'vertical_gap_threshold': 25
        }

    def setup_ocr(self):
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except Exception as e:
            logger.warning(f"OCR not available: {e}")
            self.ocr_available = False
    
    def extract_ocr_blocks(self, page, page_num: int) -> List[ContentBlock]:
        if not self.ocr_available:
            return []
        
        ocr_blocks = []        
        images = page.get_images(full=True)

        if not images:
            return []
        
        for img_index, img in enumerate(images):
            try: 
                blocks = self._process_single_image(img, page, page_num, img_index)
                ocr_blocks.extend(blocks)
            except Exception as e:
                logger.error(f"OCR failed for image {img_index} on page {page_num}: {e}")
        
        return ocr_blocks
    
    def _process_single_image(self, img, page, page_num: int, img_index: int) -> List[ContentBlock]:
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]

        with Image.open(io.BytesIO(image_bytes)) as image:
            best_result = None
            best_confidence = 0

            for config in self._get_ocr_configs():
                try:
                    enhanced_image = self._enhance_image(image.copy())
                    ocr_data = pytesseract.image_to_data(
                        enhanced_image,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )

                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        if avg_conf > best_confidence:
                            best_confidence = avg_conf
                            best_result = ocr_data
                
                except Exception as e:
                    logger.debug(f"OCR config {config} failed: {e}")
                    continue
            
            if best_result:
                return self._reconstruct_text_blocks(best_result, page_num, img_index)
            else:
                logger.warning(f"All OCR attempts failed for image {img_index} on page {page_num}")
                return []
    
    def _get_ocr_configs(self) -> List[str]:
        return [
            '--oem 3 --psm 6', # Uniform block of text
            '--oem 3 --psm 4', # Single column text
            '--oem 3 --psm 3', # Fully automatic page segmentation
            '--oem 3 --psm 1', # Automatic page segmentation with OSD
        ]
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        zoom_factor = self.config['ocr_zoom_factor']
        new_size = (int(width * zoom_factor), int(height * zoom_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        if self.config['enhance_contrast']:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(CONTRAST_ENHANCEMENT_FACTOR)
        
        image = image.convert('L')
        img_array = np.array(image)

        threshold = np.mean(img_array) - THRESHOLD_OFFSET
        img_array = np.where(img_array > threshold, PIXEL_WHITE, PIXEL_BLACK)

        return Image.fromarray(img_array.astype(np.uint8))
    
    def _group_words_into_lines(self, ocr_data: Dict) -> List[Dict]:
        words = []

        for i, text in enumerate(ocr_data['text']):
            if not text.strip():
                continue

            confidence = int(ocr_data['conf'][i])
            if confidence < self.config['ocr_confidence_threshold']:
                continue

            words.append({
                'text': text,
                'confidence': confidence,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            })
        
        lines = {}
        for word in words:
            line_key = round(word['top'] / self.config['line_height_threshold']) * self.config['line_height_threshold']
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(word)
        
        for line_words in lines.values():
            line_words.sort(key=lambda w: w['left'])
        
        line_objects = []
        for line_top in sorted(lines.keys()):
            line_words = lines[line_top]
            line_text = ' '.join(word['text'] for word in line_words)
            avg_confidence = sum(word['confidence'] for word in line_words) / len(line_words)

            line_objects.append({
                'text': line_text,
                'confidence': avg_confidence,
                'top': line_top,
                'words': line_words
            })
        
        return line_objects
    
    def _group_ocr_text(self, ocr_data: Dict, page_num: int) -> List[ContentBlock]:
        blocks = []
        lines = {}

        for i, text in enumerate(ocr_data['text']):
            if not text.strip():
                continue

            confidence = int(ocr_data['conf'][i])
            if confidence < self.config['ocr_confidence_threshold']:
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

            should_start_new = self._should_start_new_block(
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
    
    def _should_start_new_block(self, line_text: str, current_block: List, line_y: int, last_line_y: int, avg_height: float) -> bool:
        if not current_block:
            return True

        if last_line_y is not None and (line_y - last_line_y) > OCR_LINE_GROUP_HEIGHT * 2:
            return True

        content_type = classify_content(line_text, source='ocr')
        return content_type in [ContentType.HEADING, ContentType.LIST]

    
    def _create_ocr_block(self, block_lines: List, page_num: int, blocks: List):
        block_text = '\n'.join(line[0] for line in block_lines)
        avg_conf = sum(line[1] for line in block_lines) / len(block_lines)

        block = create_content_block(
            content=block_text,
            page_num=page_num,
            confidence=avg_conf / 100.0,
            source='ocr',
            metadata={'line_count': len(block_lines)},
            config=self.config
        )

        if block:
            blocks.append(block)

class PDFParser:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.text_extractor = TextExtractor(self.config.get('text_extraction', {}))
        self.ocr_extractor = OCRExtractor(self.config.get('ocr', {}))
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
                text_blocks = self._process_text_blocks(page, page_num + 1)
                content_blocks.extend(text_blocks)
            
            if has_images:
                logger.info(f"Page {page_num + 1}: Extracting images")
                ocr_blocks = self._ocr_image(page, page_num + 1)
                content_blocks.extend(ocr_blocks)

        doc.close()
        return content_blocks