# Use an official Python runtime as a parent image
#source https://saturncloud.io/blog/how-to-install-packages-with-miniconda-in-dockerfile-a-guide-for-data-scientists/
FROM debian:latest


# Install Miniconda
RUN apt-get update && apt-get install -y wget nano git g++ && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda 

# set the path
ENV PATH="/root/miniconda/bin:${PATH}"

RUN mkdir /app
RUN mkdir -p /app/model
RUN mkdir -p /app/ASVspoof2021_LA_eval
RUN mkdir -p /app/results

WORKDIR /app

# Clone the GitHub repository 
RUN git clone https://github.com/TzurV/SSL_Anti-spoofing.git 

# Create a Conda environment 
RUN conda init bash && conda create -n SSL_Spoofing python=3.7 

# Activate the Conda environment 
#RUN echo "conda init bash" >> ~/.bashrc 
RUN echo "conda activate SSL_Spoofing" >> ~/.bashrc 
SHELL ["/bin/bash", "--login", "-c"] 

RUN conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cpuonly pandas -c pytorch


RUN cd /app/SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 && pip install ./
#RUN pip install ./

RUN cd /app/SSL_Anti-spoofing && pip install tensorboard tensorboardX librosa==0.9.1
