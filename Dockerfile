# Runs the Merchant Pulse Streamlit app. Works for Hugging Face Spaces (Docker SDK, port 7860) and Cloud Run (honors $PORT).
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Spaces runs as uid 1000; data/ and registry/ must be writable for feedback, runs and retraining
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app/Home.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"]
