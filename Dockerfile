# 1. Use an official, lightweight Python environment as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your files (the model, the API code) into the container
COPY . .

# 5. Open port 8000 so the outside world can communicate with it
EXPOSE 8000

# 6. The command to start the server when the container turns on
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]