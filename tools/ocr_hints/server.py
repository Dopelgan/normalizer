"""
Сервис OCR-подсказок (DRAWING_OCR_HINTS_ENDPOINT): POST / {"image": base64 PNG} ->
{"lines": [{"text", "score", "box": [x1, y1, x2, y2] px}], "device": "gpu:0" | "cpu"}.

Устройство (OCR_DEVICE): «auto» — GPU, если свободной видеопамяти не меньше
OCR_GPU_MIN_FREE_MB (PP-OCRv5 с плитками 2000 px берёт до 3,4 ГБ), иначе CPU с
мобильным детектором (≈85 с на лист: для ночной переобработки, не на лету).
"gpu:0" и "cpu" — принудительно. Слушает OCR_PORT (8090) на всех адресах
контейнера: наружу порт не публикуется, доступ — из сети воркеров.
"""

import base64
import io
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image


def _free_gpu_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001 — нет nvidia-smi или видеокарты
        return 0


def _device() -> str:
    wanted = os.environ.get("OCR_DEVICE", "auto")
    if wanted != "auto":
        return wanted
    return "gpu:0" if _free_gpu_mb() >= int(os.environ.get("OCR_GPU_MIN_FREE_MB", "3500")) else "cpu"


DEVICE = _device()


def _engine():
    from paddleocr import PaddleOCR

    detector = "PP-OCRv5_server_det" if DEVICE.startswith("gpu") else "PP-OCRv5_mobile_det"
    return PaddleOCR(text_detection_model_name=detector, text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True,
                     text_det_limit_side_len=5000, text_det_limit_type="max", device=DEVICE,
                     enable_mkldnn=False)


OCR = _engine()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — имя метода задаёт http.server
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            image = Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")
            res = OCR.predict(np.array(image))[0]
            boxes = res["rec_boxes"].tolist() if len(res["rec_boxes"]) else []
            lines = [{"text": str(t), "score": round(float(s), 3), "box": [float(v) for v in b]}
                     for t, s, b in zip(res["rec_texts"], res["rec_scores"], boxes) if str(t).strip()]
            payload, code = {"lines": lines, "device": DEVICE}, 200
        except Exception as exc:  # noqa: BLE001 — ответ об ошибке, сервис живёт
            payload, code = {"error": str(exc)}, 500
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    # Один поток: движок Paddle не рассчитан на параллельные вызовы.
    HTTPServer(("0.0.0.0", int(os.environ.get("OCR_PORT", "8090"))), Handler).serve_forever()
