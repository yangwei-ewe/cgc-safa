# dataset_partitioner/ds_partitioner.dockerfile
FROM python:3.10-slim

RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir torchvision numpy requests

WORKDIR /app
COPY src/dataset_partitioner.py .

ENV NUM_CLIENTS=10 \
    ALPHA=0.5 \
    OUTPUT_DIR=/data

ENTRYPOINT ["python", "dataset_partitioner.py"]
