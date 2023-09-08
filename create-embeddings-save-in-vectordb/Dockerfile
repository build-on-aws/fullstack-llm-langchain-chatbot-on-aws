# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Set the AWS region environment variable
ENV AWS_DEFAULT_REGION=us-east-1

# Update pip
RUN pip install --upgrade pip

# Copy the local files into the container
COPY ./configsoa.yml /app/configsoa.yml
COPY ./startup_script.py /app/startup_script.py

# Install the required packages using pip
RUN pip install botocore boto3 langchain unstructured pdf2image pdfminer pdfminer.six

# Command to run when the container starts
CMD [ "python", "./startup_script.py"]
