FROM ubuntu:18.04

# Install required packages
RUN 	apt-get update && \
    	apt-get install -y \
		python3 \
		python3-pip \
		libgfortran4

RUN pip3 install numpy scipy pandas matplotlib

RUN python3 -m pip install pyqt5==5.14

RUN python3 -m pip install fmpy[complete]

