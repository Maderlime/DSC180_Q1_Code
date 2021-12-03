##ssh
ssh <user>@dsmlp-login.ucsd.edu

## building the docker file
docker build -t test .
docker run -it --rm mjtjoa/dsc180a_quarter1_code bash
docker run -it --rm test bash ???

docker tag test mjtjoa/dsc180a_quarter1_code
docker push mjtjoa/dsc180a_quarter1_code:latest

# Cleaning docker
docker system prune
docker system prune -a
sudo rm -rf /var/lib/docker

### Launching the Docker File
launch-scipy-ml.sh -i mjtjoa/dsc180a_quarter1_code:latest -P Always



 ld_impl_linux-64-2.3 | 667 KB    | ########## | 100%                                                
 => => # setuptools-59.4.0    | 1016 KB   | ########## | 100%                                                
 => => # zlib-1.2.11          | 86 KB     | ########## | 100%                                                
 => => # readline-8.1         | 295 KB    | ########## | 100%                                                
 => => # tk-8.6.11            | 3.3 MB    | ########## | 100%                                                
 => => # python-3.7.12     