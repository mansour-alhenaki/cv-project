# 🏭 Factory Safety System

نظام مراقبة السلامة في المصانع — مبني على YOLO + FastAPI

---

## 📁 هيكل المشروع

```
safety_system/
├── main.py                      # FastAPI entry point
├── requirements.txt
├── weights/                     # ← ضع هنا الـ .pt files
│   ├── ppe_best.pt
│   ├── id_best.pt
│   └── ladder_best.pt
├── logic/
│   ├── pipeline.py              # يدمج الثلاث مودلز
│   ├── ppe_checker.py           # منطق فحص PPE
│   ├── ladder_checker.py        # منطق السلم (زاوية + زونات + 3من4)
│   └── id_reader.py             # OCR لقراءة الـ ID
├── database/
│   └── db.py                    # SQLite - سجلات + موظفين
├── tracker/
│   └── tracker.py               # DeepSORT
└── tests/
    └── test_pipeline.py         # اختبارات
```

---

## 🚀 خطوات التشغيل

### 1. نزّل الـ weights من Google Drive
```
weights/ppe_best.pt    ← من runs/ppe_model/weights/best.pt
weights/id_best.pt     ← من runs/id_model/weights/best.pt
weights/ladder_best.pt ← من runs/ladder_model/weights/best.pt
```

### 2. ثبّت المتطلبات
```bash
pip install -r requirements.txt
```

### 3. شغّل الاختبارات أولاً
```bash
python tests/test_pipeline.py
```

### 4. شغّل الـ Backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. افتح الـ API Docs
```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | URL | الوصف |
|--------|-----|-------|
| WS | `/ws/stream` | بث الكاميرا real-time |
| WS | `/ws/analyze` | استقبال فريم من التطبيق |
| GET | `/status` | العمال النشطين الآن |
| GET | `/logs` | آخر 50 سجل |
| GET | `/logs/{employee_id}` | سجلات موظف معين |
| GET | `/employees` | قائمة الموظفين |

---

## 📦 مثال على رد الـ WebSocket

```json
{
  "total_workers": 2,
  "has_ladder": true,
  "workers": [
    {
      "track_id": 1,
      "employee_id": "EMP-042",
      "employee_name": "أحمد محمد",
      "is_safe": false,
      "ppe": {
        "compliant": false,
        "present": ["helmet", "vest", "boots"],
        "missing": ["gloves", "goggles"],
        "missing_ar": ["القفازات", "النظارات الواقية"]
      },
      "ladder": {
        "angle": 65.5,
        "angle_risk": "warning",
        "zone": "danger",
        "three_point": {
          "safe": false,
          "contact_count": 2,
          "label": "2/4"
        },
        "alerts": [
          "⚠️ زاوية السلم قريبة من الخطر (65.5°)",
          "🚨 العامل في الزون الخطر (أعلى السلم)",
          "🚨 قاعدة 3 من 4 منتهكة (2/4 نقاط ملامسة)"
        ]
      },
      "alerts": [
        {
          "type": "PPE_VIOLATION",
          "msg": "العامل EMP-042 لا يرتدي: القفازات، النظارات الواقية"
        },
        {
          "type": "LADDER_VIOLATION",
          "msg": "العامل EMP-042 - 🚨 العامل في الزون الخطر"
        }
      ]
    }
  ]
}
```

---

## ⚙️ إعداد الـ Class IDs

**مهم:** عدّل الأرقام في `logic/ppe_checker.py` حسب ترتيب كلاساتك في data.yaml:

```python
PPE_CLASSES = {
    "person":  0,   # ← تأكد هذا صح
    "helmet":  1,
    "vest":    2,
    "gloves":  3,
    "goggles": 4,
    "boots":   5,
}
```

---

## 📷 تغيير مصدر الكاميرا

في `main.py`:
```python
# كاميرا محلية
cap = cv2.VideoCapture(0)

# كاميرا RTSP
cap = cv2.VideoCapture("rtsp://192.168.1.100:554/stream")

# فيديو للاختبار
cap = cv2.VideoCapture("test_video.mp4")
```
