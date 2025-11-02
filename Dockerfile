# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code (API, model, data)
COPY . .

# Expose API port (if using FastAPI/Flask)
EXPOSE 8080

# Run the app
CMD ["python", "api.py"]
