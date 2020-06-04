FROM python:3.7

# Form directories
RUN mkdir /code
RUN mkdir /output
RUN mkdir /data

# Get data
RUN wget http://cmp.felk.cvut.cz/\~chum/XEP33SAM/sift.zip
RUN unzip sift.zip
RUN mv SIFT.dat data
RUN rm sift.zip
RUN wget http://cmp.felk.cvut.cz/~chum/XEP33SAM/CNNdesc105k.dat
RUN mv CNNdesc105k.dat data

# Add code
ADD cluster.py /code
ADD task4_run.py /code
ADD flann_translation.py /code

# Install libraries
RUN pip install numpy faiss-gpu==1.6 pandas pyflann 2to3 scikit-learn

# Set up the working directory
WORKDIR /code
RUN python flann_translation.py

# Run
CMD [ "python", "./task4_run.py" ]
