import re
from wordsegment import segment, load

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
