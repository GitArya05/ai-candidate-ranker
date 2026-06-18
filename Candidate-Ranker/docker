# Use an official lightweight Python runtime
FROM python:3.11-slim

# Install core Linux utilities required for compiling models like FAISS or LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a clean non-root user account for Hugging Face security compliance
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create app directory
WORKDIR $HOME/app

# Copy requirements file first to leverage Docker layer caching
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Upgrade setup tools and install your dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyterlab

# Download your required spaCy NLP vocabulary models
RUN python -m spacy download en_core_web_sm

# Copy the rest of your source code and data assets into the container
COPY --chown=user src/ $HOME/app/src/
COPY --chown=user data/ $HOME/app/data/

# Expose port 7860 for Hugging Face Web Access
EXPOSE 7860

# Start JupyterLab on the specific port required by Hugging Face
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=7860", "--no-browser", "--AllowRoot", "--NotebookApp.token=''", "--NotebookApp.password=''"]