"""
logic/id_reader.py
------------------
يقرأ رقم الـ ID من صورة الـ badge
"""

import easyocr
import numpy as np
import re

# أنشئ الـ reader مرة وحدة (بطيء في الأول)
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        print("🔄 Loading EasyOCR...")
        import torch
        use_gpu = torch.cuda.is_available()
        print(f"🔄 EasyOCR - GPU: {use_gpu}")
        _reader = easyocr.Reader(['en', 'ar'], gpu=use_gpu)
        print("✅ EasyOCR ready")
    return _reader


def read_id_from_frame(frame: np.ndarray, badge_bbox: list) -> str | None:
    """
    frame      : الفريم الكامل
    badge_bbox : [x1, y1, x2, y2] موقع الـ badge

    Returns: نص الـ ID مثل "EMP-042" أو None
    """
    x1, y1, x2, y2 = [int(v) for v in badge_bbox]

    # padding عشان ما نقطع الحروف
    pad = 15
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    reader  = _get_reader()
    results = reader.readtext(crop)

    if not results:
        return None

    # رتّب حسب الـ confidence واختار الأعلى
    results.sort(key=lambda r: r[2], reverse=True)

    for _, text, conf in results:
        if conf < 0.4:
            continue
        cleaned = _clean_id(text)
        if cleaned:
            return cleaned

    return None


def _clean_id(text: str) -> str | None:
    """
    نظّف النص وتأكد أنه يشبه ID
    يقبل: EMP-042 / emp042 / 042 / ID-5
    """
    text = text.upper().strip().replace(" ", "")

    # لو فيه حروف وأرقام = ID محتمل
    if re.search(r'[A-Z0-9\-]{3,}', text):
        return text

    return None
