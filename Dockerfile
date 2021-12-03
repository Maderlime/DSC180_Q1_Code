# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

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

######################################################
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt-get install -y mecab-ipadic-utf8

ENV HOME="/jovyan"

WORKDIR $HOME
RUN apt-get install -y git

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.7.11
RUN pyenv global 3.7.11
RUN exec $SHELL
RUN pip install wheel==0.37.0
RUN pip install numpy==1.19.5
RUN pip install tensorflow==1.15
RUN pip install tensorboard==1.15.0
RUN pip install matplotlib==3.4.2
RUN pip install scipy==1.7.1
RUN pip install Pillow==8.3.1 

######################################################
# WORKDIR /home/python_user
# # USER python_user

# RUN apt-get -y install git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl 
# RUN git clone git://github.com/yyuu/pyenv.git .pyenv

# ENV HOME  /home/python_user
# ENV PYENV_ROOT $HOME/.pyenv
# ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# RUN pyenv install 3.7.11
# RUN pyenv global 3.7.11
# RUN pyenv rehash

###################################################

# RUN pip install virtualenvwrapper

# RUN git clone https://github.com/yyuu/pyenv.git ~/.pyenv
# RUN git clone https://github.com/yyuu/pyenv-virtualenvwrapper.git ~/.pyenv/plugins/pyenv-virtualenvwrapper

# RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
# RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
# RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
# RUN echo 'pyenv virtualenvwrapper' >> ~/.bashrc

# RUN exec $SHELL

######################################################

# RUN apt-get -y install htop make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
# RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
# RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
# RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
# RUN exec "$SHELL"
# # RUN curl https://pyenv.run | bash




# 3) install packages using notebook user
USER jovyan
# RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# RUN apt-get install -y mecab-ipadic-utf8

# ENV HOME="/root"

# WORKDIR $HOME
# RUN apt-get install -y git

# RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
# ENV PYENV_ROOT="$HOME/.pyenv"
# ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
# RUN pyenv install 3.7.11
# RUN pyenv global 3.7.11
# RUN exec $SHELL

# git clone https://github.com/pyenv/pyenv.git ~/.pyenv
# COPY --from=compile-image --chown=jovyan /opt/venv .
# RUN pip install wheel==0.37.0
# RUN pip install numpy==1.19.5
# RUN pip install tensorflow==1.15
# RUN pip install tensorboard==1.15.0
# RUN pip install matplotlib==3.4.2
# RUN pip install scipy==1.7.1
# RUN pip install Pillow==8.3.1 
# RUN conda install -y scikit-learn==0.24.2
# RUN conda config --set allow_conda_downgrades true
# RUN conda install -n root conda=4.10.3

# RUN conda config --append channels conda-forge
# RUN conda config --append channels pkgs/main
# RUN conda create -n myenv python=3.7
# RUN conda install -n myenv wheel==0.37.0
# RUN conda install -n myenv numpy==1.19.5
# RUN conda install -n myenv tensorflow==1.15
# RUN conda install -n myenv -c conda-forge tensorboard==1.15.0
# RUN conda install -n myenv -c conda-forge matplotlib==3.4.2  scipy==1.7.1 Pillow==8.3.1 
# RUN conda init bash



# COPY requirements.txt requirements.txt

# RUN pip install --user -r requirements.txt

# RUN pip install --no-cache-dir matplotlib numpy==1.19.5 scipy Pillow
# RUN pip install tensorflow==1.5.1

# RUN pip install --no-cache-dir matplotlib numpy scipy Pillow tensorflow
# RUN pip install tensorflow


# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
