FROM python:3.7.5-slim
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
ADD requirements.txt .
WORKDIR .
RUN pip install -r requirements.txt
ADD . .
ENTRYPOINT ["python"]
CMD ["app.py"]
