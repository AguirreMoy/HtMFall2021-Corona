# Your starting image source
FROM python:3.8-slim-buster

# Working directory
WORKDIR /app
# COPY our /app files into the image and requirements
COPY /app /app
COPY requirements.txt requirements.txt

#Installing dependencies # Install any Python dependecies we need

RUN apt-get update && apt-get install libgomp1 -y  && pip3 install -r requirements.txt && pip3 install /app/ember-master.zip

# Command our Docker container will run to generate a submission CSV
CMD ["python3", "/app/my_testing_script.py"]




