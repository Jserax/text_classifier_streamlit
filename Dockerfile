FROM python:3.9-slim

RUN mkdir /app
COPY . app/
WORKDIR /app

RUN pip install -r requirements.txt

RUN python3 prepare_data.py
RUN python3 train_model.py
