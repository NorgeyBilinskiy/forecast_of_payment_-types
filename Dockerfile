FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install pymorphy3-dicts-ru

COPY . .

RUN touch /app/first_run.flag

CMD ["sh", "-c", "if [ ! -f /app/first_run.flag ]; then python __main__.py && touch /app/first_run.flag; fi"]
