from .text_segmentation import text_segmentation_class
# Initialize Tokenizer  

def process_pdf(text):
    
    seg_obj = text_segmentation_class(text, '../media/glove.6B.100d.txt')
    print(seg_obj)
    return seg_obj.doc
    
    segments = segment_text(text)  
    final_segments = []  
    for segment in segments:  
        parts = ensure_token_limit(segment)  
        final_segments.extend(parts)  
    return final_segments  