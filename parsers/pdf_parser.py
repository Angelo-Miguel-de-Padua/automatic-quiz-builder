import fitz
import logging
import io
import re
import os
import json
from paddleocr import PaddleOCR
from wordsegment import segment
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

class TextProcessor:
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
        return " ".join(self.segment_with_punct(text))
    
    def segment_with_punct(self, text):
        tokens = re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)
        result = []
        for token in tokens:
            if re.match(r"[A-Za-z0-9]+", token):
                segmented = segment(token.lower())
                idx = 0
                for word in segmented:
                    part = token[idx:idx+len(word)]
                    result.append(part)
                    idx += len(word)
            else:
                result.append(token)
        return result

class ImageProcessor:
    def enhance_image(self, img):
        img = img.convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
        return img
    
    def resize_if_needed(self, img, max_dim=2500):
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return img
    
    def save_debug_image(self, img, page_num, output_dir="ocr_inputs"):
        os.makedirs(output_dir, exist_ok=True)
        img.save(f"{output_dir}/page_{page_num}.png")
    
class OCRProcessor:
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(lang=lang)
        self.text_processor = TextProcessor()
    
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
                            cleaned_text = self.text_processor.clean_ocr_text(text.strip())
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
    
    def _extract_bbox(self, bboxes, index):
        if index < len(bboxes) and bboxes[index] is not None:
            poly = bboxes[index]
            if isinstance(poly[0], (list, np.ndarray)):
                x_coords = [point[0] for point in poly]
                y_coords = [point[1] for point in poly]
            else:
                x_coords = [poly[i] for i in range(0, len(poly), 2)]
                y_coords = [poly[i] for i in range(1, len(poly), 2)]
            return [
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords))
            ]
        else:
            return [0, index * 20, 200, (index + 1) * 20]

class DataFormatter:
    def format_plain_text(self, pages_data):
        formatted_text = []
        for page_data in pages_data:
            formatted_text.append(f"\n=== Page {page_data['page']} ({page_data['source']}) ===\n")
            text = page_data["text"].strip()
            text = re.sub(r"[ \t]+", " ", text)
            formatted_text.append(text + "\n")
        return "".join(formatted_text)

class OutputManager:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_plain_text(self, formatted_text, base_name, custom_path=None):
        output_path = custom_path or os.path.join(self.output_dir, f"{base_name}_extracted.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"Saved plain text: {output_path}")
        return output_path
        
    def save_structured_json(self, pages_data, base_name, custom_path=None):
        output_path = custom_path or os.path.join(self.output_dir, f"{base_name}_structured.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, indent=2, ensure_ascii=False)
        print(f"Saved structured JSON: {output_path}")
        return output_path
    
class PDFParser:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.text_extractor = TextProcessor(self.config.get('text_processor', {}))
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