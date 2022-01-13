FROM python:3.7
COPY . .
WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT [ "executable" ]
#CMD ["gunicorn", "-w 4", "main:app"]