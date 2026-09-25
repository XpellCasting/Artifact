FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-gat.txt requirements-gat-docker.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_GAT=0
RUN if [ "$INSTALL_GAT" = "1" ]; then pip install --no-cache-dir -r requirements-gat-docker.txt; fi

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
