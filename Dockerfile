# Use an official Python runtime as a parent image
#source https://saturncloud.io/blog/how-to-install-packages-with-miniconda-in-dockerfile-a-guide-for-data-scientists/
FROM debian:latest


# Install Miniconda
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda 

# set the path
ENV PATH="/root/miniconda/bin:${PATH}"


# Clone the GitHub repository 
#RUN git clone https://github.com/TzurV/SSL_Anti-spoofing.git 

# Create a Conda environment 
RUN conda init bash && conda create -n SSL_Spoofing python=3.7 

# Activate the Conda environment 
#RUN echo "conda init bash" >> ~/.bashrc 
RUN echo "conda activate SSL_Spoofing" >> ~/.bashrc 
SHELL ["/bin/bash", "--login", "-c"] 

#RUN pip install torch==1.8.1+cu111 
#torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
