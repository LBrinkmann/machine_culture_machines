## Install docker

## Setup Docker

make docker being usable without sudo

```
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo service docker restart
newgrp docker
```

login to docker

```
docker login -p <pw> -u <user>
```

## Create Docker Image

Build, tag, push image

```
docker build -f deployment/Dockerfile.mb .
docker tag cc5e9513e72c levinb/machine-culture:0.2.1
docker push levinb/machine-culture:0.2.1
```

##
