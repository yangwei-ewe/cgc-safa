FROM python:3.10.16-slim-bullseye AS kubeflower
WORKDIR /app
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1000 --retries 20 -r requirements.txt
COPY src/ src/
ENV max_tries=60
ENV period=2
CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]
