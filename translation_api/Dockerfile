# Use a Python base image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Copy the requirements file and application code
COPY requirements.txt ./
COPY translation_api.py ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "translation_api:app", "--host", "0.0.0.0", "--port", "8000"]