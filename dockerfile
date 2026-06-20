FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt $HOME/app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyterlab

RUN python -m spacy download en_core_web_sm

COPY --chown=user src/ $HOME/app/src/
COPY --chown=user data/ $HOME/app/data/

EXPOSE 7860

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=7860", "--no-browser", "--AllowRoot", "--NotebookApp.token=''", "--NotebookApp.password=''"]
