import fitz
import logging
import io
import re
import os
import json
import cv2
from paddleocr import PaddleOCR
from wordsegment import segment, load
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEXT_DENSITY_MAX_THRESHOLD = 80
TEXT_DENSITY_DYNAMIC_RATIO = 0.7
MIN_CONTOUR_AREA = 10

class TextCleaner:
    def __init__(self):
        load()
    
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

class ImageAnalyzer:
    def _calculate_text_density(gray_array: np.ndarray) -> float:
        mean_brightness = np.mean(gray_array)
        threshold = min(TEXT_DENSITY_MAX_THRESHOLD, mean_brightness * TEXT_DENSITY_DYNAMIC_RATIO)
        return np.sum(gray_array <= threshold) / gray_array.size
    
    def _estimate_text_height(gray_array: np.ndarray) -> float:
        _, binary = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        heights = [cv2.boundingRect(contours)[3] for contour in contours
                   if cv2.contourArea(contour) > MIN_CONTOUR_AREA]
        
        return np.median(heights) if heights else 0
    
    def _calculate_background_uniformity(image: Image.Image) -> float:
        gray_array = np.image(image.convert("L"))
        height, width = gray_array.shape

        local_std_blocks = []
        block_size = 32

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = gray_array[y:y+block_size, x:x+block_size]
                if block_size > 0:
                    local_std_blocks.append(np.std(block))
        
        return np.mean(local_std_blocks) if local_std_blocks else 0
    
    def _determine_contrast_target(text_density: float) -> int:
        if text_density > 0.2:
            return 38   # high density
        elif text_density > 0.1:
            return 42   # medium density
        else:
            return 46   # low density 

class ImageProcessor:
    def enhance_image(self, img):
        img = img.convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
        img = ImageOps.expand(img, border=10, fill='white')
        return img

    def resize_if_needed(self, img, max_dim=2500):
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return img
    
    def save_debug_image(self, img, page_num, output_dir="ocr_inputs"):
        os.makedirs(output_dir, exist_ok=True)
        img.save(f"{output_dir}/page_{page_num}.png")
    
class OCREngine:
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(lang=lang)
        self.text_processor = TextCleaner()   
    
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

                    for j, (text, score) in enumerate(zip(texts, scores)):
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

class TextFormatter:
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
    def __init__(self, ocr_lang="en", output_dir="outputs"):
        self.ocr_processor = OCREngine(ocr_lang)
        self.image_processor = ImageProcessor() 
        self.data_formatter = TextFormatter()
        self.output_manager = OutputManager(output_dir)
        self.doc = None 

    def extract_from_pdf(self, pdf_path):
        self.doc = fitz.open(pdf_path)  
        try:
            for i in range(len(self.doc)):
                page = self.doc[i]
                txt = page.get_text("text").strip()
                has_text = bool(txt)
                has_images = len(page.get_images(full=True)) > 0

                combined_text = txt
                text_bbox_pairs = []

                if has_text:
                    print(f"[TEX] Page {i+1}: {len(txt)} chars (has image? {has_images})")
                    page_rect = page.rect
                    text_bbox_pairs.append({
                        'text': txt,
                        'bbox': [0, 0, int(page_rect.width), int(page_rect.height)]
                    })

                if has_images:
                    print(f"[IMG] Page {i+1} has {len(page.get_images(full=True))} images -> OCRing them")
                    image_pairs = self._process_page_images_with_ocr(page, i + 1)
                    combined_text += "\n" + " ".join([pair['text'] for pair in image_pairs])
                    text_bbox_pairs.extend(image_pairs)

                if has_text or has_images:
                    yield {
                        'page': i + 1,
                        'text': combined_text.strip(),
                        'source': 'text+image' if has_text and has_images else ('text' if has_text else 'ocr'),
                        'bbox': [0, 0, int(page.rect.width), int(page.rect.height)],
                        'text_bbox_pairs': text_bbox_pairs
                    }
                else:
                    print(f"[EMPTY] Page {i+1} has no text and no image")
        finally:
            self.doc.close()
            self.doc = None  

    def _process_page_images_with_ocr(self, page, page_num):
        image_pairs = []
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = self.doc.extract_image(xref)  
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))
                img_pil = self.image_processor.enhance_image(img_pil)
                img_pil = self.image_processor.resize_if_needed(img_pil)
                self.image_processor.save_debug_image(img_pil, f"{page_num}_{img_index}", output_dir="ocr_inputs/page_images")
                img_arr = np.array(img_pil)
                ocr_result = self.ocr_processor.process_image(img_arr, page_num)
                image_pairs.extend(ocr_result.get("text_bbox_pairs", []))
            except Exception as e:
                print(f"Failed to OCR image {img_index} on page {page_num}: {e}")
        return image_pairs
    
    def process_pdf(self, pdf_path, save_all_formats=True):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pages_data = list(self.extract_from_pdf(pdf_path))

        results = {
            'pages_data': pages_data,
            'base_name': base_name,
            'files_saved': {}
        }

        if save_all_formats:
            structured_path = self.output_manager.save_structured_json(pages_data, base_name)
            results['files_saved']['structured_json'] = structured_path

            formatted_text = self.data_formatter.format_plain_text(pages_data)
            text_path = self.output_manager.save_plain_text(formatted_text, base_name)
            results['files_saved']['plain_text'] = text_path

        return results

def main():
    pdf_file = "test_files/LISN09L Topic 7 (Revised).pdf"
    extractor = PDFParser(ocr_lang="en", output_dir="outputs")
    print("\nProcessing PDF and saving all formats...")
    results = extractor.process_pdf(pdf_file, save_all_formats=True)
    print("\nFiles saved:")
    for format_name, file_path in results['files_saved'].items():
        print(f"  {format_name}: {file_path}")


if __name__ == "__main__":
    main()