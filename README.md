# STEPS TO RUN CODE ON ADVERSARIAL ROBUST TRAINING (QUARTER 2 - DSC 180B)

## How to SSH into the DSMLP server
ssh [user]@dsmlp-login.ucsd.edu

## How to build the Docker file
docker build -t test .
docker run -it --rm test /bin/bash
  
## Deploy a pod with GPU support
launch-scipy-ml-gpu.sh

# LOADING IN DATA

Load in the data from the following source: https://www.dropbox.com/sh/tg6xij9hhfzgio9/AADqu6BMq3Rko7U7-q6vwmMFa?dl=0

We will use the following files for each dataset:
- val_test_x_preprocess.npy
- val_test_y.npy

Make a folder in for the test and train data for each dataset. Within each of these folders, create two subfolders titled as '0' and '1'. 

From here, go to the file at DSC180_Q1_Code/patch_attacks/data/cxr/make_fast_adversarial_documents.ipynb and run the code in these cells. This will load in the images as Numpy files and partition them into training and test sets. Set the output writing dirctories to the folders you created above. Use a 70/30 split in the ranges in the code based on the size of the dataset. 

# MAKING ADJUSTMENTS TO THE CODE

You can edit hyperparameters for the FGSM training model in the train_fgsm.py file in src/test. You can edit attack parameters in the evaluate_pgd method on the utils.py file in src/model. You can select the datasets you want to load in from the ones you created above. 


# Command line prompt to run the code
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
