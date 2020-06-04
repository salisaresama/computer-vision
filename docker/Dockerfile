FROM python:3.7

# Form directories
RUN mkdir /code
RUN mkdir /output
RUN mkdir /data

# Get data
RUN wget http://cmp.felk.cvut.cz/\~chum/XEP33SAM/sift.zip -P /data/
RUN tar -xf data/sift.zip -C data/

# Add code
ADD cluster.py /code
ADD task4_run.py /code
ADD flann_translation.py /code

# Install libraries
RUN pip install numpy faiss-gpu pandas pyflann 2to3 scikit-learn

# Set up the working directory
WORKDIR /code
RUN python flann_translation.py

# Run
CMD [ "python", "./task4_run.py" ]
