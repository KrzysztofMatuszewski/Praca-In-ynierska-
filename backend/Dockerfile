FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./lckc-ssl.crt /code/lckc-ssl.crt

RUN pip --cert /code/lckc-ssl.crt install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# CMD ["fastapi", "run", "app/main.py", "--port", "8009","--workers", "4"]
CMD ["uvicorn", "app.main:app", "--host", "172.19.20.25", "--port", "8009", "--workers", "8"] 