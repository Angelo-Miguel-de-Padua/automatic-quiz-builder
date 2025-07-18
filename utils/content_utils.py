import re
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

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

LIST_ITEM_REGEX = re.compile(r'^\s*[\d\-\•\*]')
BOLD_FONT_FLAG = 1 << 4
MAX_WORDS_FOR_HEADING = 10

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

def classify_content(text: str, source: str = 'ocr', font_sizes: Optional[List[float]] = None, 
                    font_flags: Optional[List[int]] = None, config: Optional[dict] = None) -> str:

    if config is None:
        config = {
            'heading_font_threshold': 14,
            'bold_font_flag': BOLD_FONT_FLAG,
            'max_words_for_heading': MAX_WORDS_FOR_HEADING
        }
    
    text_lower = text.lower().strip()
    
    if source == 'text':
        if font_sizes and sum(font_sizes) / len(font_sizes) > config.get('heading_font_threshold', 14):
            return ContentType.HEADING
        
        if font_flags and any(flag & config.get('bold_font_flag', BOLD_FONT_FLAG) for flag in font_flags):
            if len(text.split()) < config.get('max_words_for_heading', MAX_WORDS_FOR_HEADING):
                return ContentType.HEADING
    
    if looks_like_heading(text, config):
        return ContentType.HEADING
    
    if looks_like_list_item(text) or ('\n' in text and any(looks_like_list_item(line) for line in text.split('\n'))):
        return ContentType.LIST
    
    if any(sym in text for sym in ['∑', '∫', '=', '≠', '√']):
        return ContentType.FORMULA
    
    if 'def ' in text_lower or 'class ' in text_lower or 'return ' in text_lower:
        return ContentType.CODE
    
    if any(phrase in text_lower for phrase in ['is defined as', 'refers to', 'means that']):
        return ContentType.DEFINITION
    
    return ContentType.PARAGRAPH

def looks_like_heading(text: str, config: Optional[dict] = None) -> bool:
    if config is None:
        config = {'max_words_for_heading': MAX_WORDS_FOR_HEADING}
    
    words = text.split()
    max_words = config.get('max_words_for_heading', MAX_WORDS_FOR_HEADING)
    
    return (len(words) <= max_words and 
            (text.isupper() or text.istitle()) and 
            not text.lower().startswith(('the ', 'and ', 'or ', 'but ', 'in ', 'on ', 'at ')))

def looks_like_list_item(text: str) -> bool:
    return (text.strip().startswith(('•', '●', '·', '-', '*')) or 
            re.match(r'^\s*\d+[\.\)]\s', text))

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
    
    return text.strip()

def group_words_into_lines(ocr_data: Dict, config: Dict) -> List[Dict]:
    words = []

    for i, text in enumerate(ocr_data['text']):
        if not text.strip():
            continue

        confidence = int(ocr_data['conf'][i])
        if confidence < config['ocr_confidence_threshold']:
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
        line_key = round(word['top'] / config['line_height_threshold']) * config['line_height_threshold']
        lines.setdefault(line_key, []).append(word)

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

def create_block_from_lines(
    lines: List[Dict], page_num: int, img_index: int, config: Dict
) -> Optional[ContentBlock]:
    if not lines:
        return None

    if len(lines) == 1:
        block_text = lines[0]['text']
    else:
        if any(looks_like_list_item(line['text']) for line in lines):
            block_text = '\n'.join(line['text'] for line in lines)
        else:
            block_text = ' '.join(line['text'] for line in lines)

    block_text = clean_text(block_text)
    if len(block_text.strip()) < config['min_text_length']:
        return None

    avg_confidence = sum(line['confidence'] for line in lines) / len(lines)
    content_type = classify_content(block_text, source='ocr')

    return ContentBlock(
        content=block_text,
        content_type=content_type,
        page_number=page_num,
        confidence=avg_confidence / 100.0,
        metadata={
            'source': 'ocr',
            'image_index': img_index,
            'line_count': len(lines),
            'avg_confidence': avg_confidence
        }
    )

def group_lines_into_blocks(
    lines: List[Dict], page_num: int, img_index: int, config: Dict
) -> List[ContentBlock]:
    blocks = []
    current_block_lines = []

    for i, line in enumerate(lines):
        should_start_new_block = False

        if i == 0:
            should_start_new_block = True
        else:
            prev_line = lines[i-1]
            vertical_gap = line['top'] - prev_line['top']

            if vertical_gap > config['vertical_gap_threshold']:
                should_start_new_block = True
            elif looks_like_heading(line['text']):
                should_start_new_block = True
            elif looks_like_list_item(line['text']):
                should_start_new_block = True

        if should_start_new_block and current_block_lines:
            block = create_block_from_lines(current_block_lines, page_num, img_index, config)
            if block:
                blocks.append(block)
            current_block_lines = []

        current_block_lines.append(line)

    if current_block_lines:
        block = create_block_from_lines(current_block_lines, page_num, img_index, config)
        if block:
            blocks.append(block)

    return blocks

def create_content_block(content: str, page_num: int, confidence: float, source: str = 'text', metadata: dict = None, config: dict = None) -> Optional[ContentBlock]:
    cleaned_content = clean_text(content)
    if not cleaned_content or len(cleaned_content) < 2:
        return None
    
    content_type = classify_content(
        cleaned_content,
        source=source,
        font_sizes=metadata.get('font_sizes') if metadata else None,
        font_flags=metadata.get('font_flags') if metadata else None,
        config=config
    )
    
    base_metadata = {'source': source}
    if metadata:
        base_metadata.update(metadata)
    
    return ContentBlock(
        content=cleaned_content,
        content_type=content_type,
        page_number=page_num,
        confidence=confidence,
        metadata=base_metadata
    )