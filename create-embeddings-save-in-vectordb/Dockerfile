# Use the official Python 3.10.13 image as the base image
FROM python:3.10.13

# Set the working directory inside the container
WORKDIR /app

# Set the AWS region environment variable
ENV AWS_DEFAULT_REGION=us-east-1

# Update pip
RUN pip install --upgrade pip

# Copy the local files into the container
COPY ./startup_script.py /app/startup_script.py

# Install the required packages using pip
RUN pip install --no-cache-dir  botocore==1.31.44 boto3==1.28.44 langchain==0.0.285 unstructured==0.10.12 pdf2image==1.16.3 pdfminer==20191125 pdfminer.six==20221105 

# Command to run when the container starts
CMD [ "python", "./startup_script.py"]
