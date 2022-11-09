FROM python:3.9

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/grzsko/MassSinkhornmetry.git
RUN git clone https://github.com/grzsko/Alignstein.git

WORKDIR MassSinkhornmetry
RUN pip install -r requirements.txt && python3 setup.py install

WORKDIR ../Alignstein
RUN pip install -r requirements.txt && python3 setup.py install

WORKDIR ~
