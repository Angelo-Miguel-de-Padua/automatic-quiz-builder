import fitz
import logging
import io
import pytesseract
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
                lines = group_words_into_lines(best_result, self.config)
                blocks = group_lines_into_blocks(lines, page_num, img_index, self.config)
                return blocks
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
                text_blocks = self.text_extractor._process_text_blocks(page, page_num + 1)
                content_blocks.extend(text_blocks)
            
            if has_images:
                logger.info(f"Page {page_num + 1}: Extracting images")
                ocr_blocks = self.ocr_extractor.extract_ocr_blocks(page, page_num + 1)
                content_blocks.extend(ocr_blocks)

        doc.close()
        return content_blocks