FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt

CMD ["python","main.py","--input_dir=/input","--predict_dir=/predict"]