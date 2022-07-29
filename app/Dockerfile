# alpine is less incase of memmory
# FROM python:3.8.13-alpine3.16
FROM python:3.8-buster

WORKDIR /user/src/app

COPY app.py ./app.py 

COPY requirements.txt .

# if you had problem with pip install --upgrade pip
# you can uncomment lines below
# RUN set -xe \
#     && apt-get update \
#     && apt-get install python3-pip
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]