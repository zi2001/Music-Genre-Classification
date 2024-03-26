FROM python:3.10.6-buster

WORKDIR /prod
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY gmc gmc
COPY setup.py setup.py
RUN pip install .

RUN apt-get update && apt-get --yes install libsndfile-dev
COPY model model

CMD uvicorn gmc.api:app --host 0.0.0.0 --port $PORT
