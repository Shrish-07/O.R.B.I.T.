# FIX C11: esda==2.10.0 requires Python >= 3.12 (CI was bumped to 3.13 for
# exactly this reason); python:3.11-slim could never install the stack.
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
