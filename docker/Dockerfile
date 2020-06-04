FROM python:3.7

# Form directories
RUN mkdir /code
RUN mkdir /output

# Add code
ADD pylib /code
ADD faiss.py /code

# Install libraries
#RUN pip install numpy faiss-gpu pandas

# Set up the working directory
WORKDIR /code

RUN ls

# Run
CMD [ "python", "./faiss.py" ]
