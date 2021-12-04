# Build Overview
### Command line prompt to run the code
#### python run.py test


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
