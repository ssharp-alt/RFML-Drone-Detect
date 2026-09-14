# Build from the project root:
#   docker build -f serving/Dockerfile -t drone-rf-classifier:latest .
# Requires checkpoints/model_scripted.pt and checkpoints/class_names.json
# to already exist (i.e. run `python -m src.train` to completion first).
FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/serving/requirements.txt

# Only the shared preprocessing/config modules -- not the training-only
# pieces (remote_zip, shard_cache, dataset, train) or their extra deps.
COPY src/__init__.py src/config.py src/transforms.py /app/src/
COPY serving/predictor.py /app/predictor.py

# License verification module ONLY -- deliberately NOT generate_license.py,
# which holds the vendor-side issuing logic and must never ship to a
# customer's image (it doesn't touch the private key file itself, but
# there's no reason to expose the issuing code either).
RUN mkdir -p /app/license
COPY license/__init__.py license/verify.py /app/license/

RUN mkdir -p /app/model
COPY checkpoints/model_scripted.pt checkpoints/class_names.json /app/model/

ENV PYTHONPATH=/app
EXPOSE 8080
ENTRYPOINT ["python", "predictor.py"]
