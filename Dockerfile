FROM python:3.9
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
WORKDIR /app
COPY . .
RUN pip install pipenv
RUN pipenv lock -r > requirements.txt && pip install -r requirements.txt 
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT