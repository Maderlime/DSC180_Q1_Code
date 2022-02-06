# STEPS TO RUN CODE ON ADVERSARIAL ROBUST TRAINING (QUARTER 2 - DSC 180B)

## How to SSH into the DSMLP server
ssh <user>@dsmlp-login.ucsd.edu

## How to build the Docker file
docker build -t test .
docker run -it --rm test /bin/bash
  
## Deploy a pod with GPU support
launch-scipy-ml-gpu.sh

## Command line prompt to run the code
python run.py test

# STEPS TO RUN CODE ON ADVERSARIAL ATTACKS (QUARTER 1 - DSC 180A)

# Build Overview
### Command line prompt to run the code
#### python run.py test_q1


## How to SSH into the DSMLP server
ssh <user>@dsmlp-login.ucsd.edu

## How to build the Docker file
docker build -t test .
docker run -it --rm test /bin/bash

<!-- docker run -it --rm mjtjoa/dsc180a_quarter1_code bash -->


docker tag test mjtjoa/dsc180a_quarter1_code
docker push mjtjoa/dsc180a_quarter1_code:latest

# Cleaning docker
docker system prune
docker system prune -a
sudo rm -rf /var/lib/docker

### Launching the Docker File
launch-scipy-ml.sh -i mjtjoa/dsc180a_quarter1_code:latest -P Always
  
### Relevant Links
  Project Report: https://docs.google.com/document/d/1iQ0lZ_wpxqQXRwwjwKANrwNn6FRDwvasISHf9vpNIHw/edit?usp=sharing
  
  Project Proposal for Q2: https://docs.google.com/document/d/1d4Z4yS0aSyCMxht0NaaEf_mMxH5KFPg2B8b3rKXyRwk/edit?usp=sharing
  
  Source Report: https://arxiv.org/pdf/1804.05296.pdf
  
  Source Repository: https://github.com/sgfin/adversarial-medicine
