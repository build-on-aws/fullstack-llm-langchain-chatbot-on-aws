# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set the AWS region environment variable
ENV AWS_DEFAULT_REGION=us-east-1

# Update pip
RUN pip install --upgrade pip


# Install the required packages using pip
RUN pip install boto3 langchain flask gunicorn

# Make port 80 available to the world outside this container
EXPOSE 80

# Command to run Gunicorn and start the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
