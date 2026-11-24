# Сервис OCR-подсказок для DRAWING_OCR_HINTS: PP-OCRv5 (PaddleOCR, Apache-2.0).
# Модели — каталог /root/.paddlex (официальные PP-OCRv5_server_det, PP-OCRv5_mobile_det,
# cyrillic_PP-OCRv5_mobile_rec, PP-LCNet_x1_0_textline_ori), смонтированный при запуске.
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --no-cache-dir "paddleocr==3.3.*" pillow
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True HF_HUB_OFFLINE=1
COPY server.py /srv/server.py
EXPOSE 8090
CMD ["python", "/srv/server.py"]
