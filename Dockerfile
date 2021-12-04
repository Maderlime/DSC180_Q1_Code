# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.1-stable

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2021.2-stable

# scipy/machine learning (tensorflow, pytorch)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt-get update
# ibncursesw5-dev 
RUN apt-get -y install htop
######################################################
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt-get install -y mecab-ipadic-utf8

# Downloading directly to jovyan, since we need admin access for doing the python 3.8->3.7 magic
ENV HOME="/jovyan"

WORKDIR $HOME
RUN apt-get install -y git

# getting the pyenv and changing it to 3.7 since this is what we need to run our python code.
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.7.11
RUN pyenv global 3.7.11

# Restarting so we can have python3.7
RUN exec $SHELL

# Installing our required files 
RUN pip install --no-cache-dir wheel==0.37.0
RUN pip install --no-cache-dir numpy==1.19.5
RUN pip install --no-cache-dir tensorflow==1.15
RUN pip install --no-cache-dir tensorboard==1.15.0
RUN pip install --no-cache-dir matplotlib==3.4.2
RUN pip install --no-cache-dir scipy==1.7.1
RUN pip install --no-cache-dir Pillow==8.3.1 h5py==2.10 


# 3) install packages using notebook user
USER jovyan


# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
