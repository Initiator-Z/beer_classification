FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY ./app /app
COPY ./models /models
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]