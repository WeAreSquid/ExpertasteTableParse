# python base image in the container from Docker Hub
FROM python:3.8.2

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set the working directory in the container to be /app
WORKDIR /app

COPY . /app

# install the packages from the Requirements.txt in the container
RUN pip3 install -r ./requirements.txt

ENV PYTHONUNBUFFERED=1

# expose the port that uvicorn will run the app on
#ENV PORT=4100
EXPOSE 4100

# execute the command python main.py (in the WORKDIR) to start the app
CMD ["python", "/app/main.py"]