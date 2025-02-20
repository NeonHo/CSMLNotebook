# Switch images
```
sudo vi /etc/apt/sources.list 

```
replace the origin text with follows:
```
deb https://mirrors.tuna.tsinghua.edu.cn/kali kali-rolling main non-free contrib non-free-firmware
# deb-src https://mirrors.tuna.tsinghua.edu.cn/kali kali-rolling main non-free contrib non-free-firmware
```
# Install Docker
```
sudo apt update

sudo apt-get install -y docker.io

sudo apt-get install docker-compose


sudo docker -v
sudo docker-compose -v 
```

Docker change images:
```
vi /etc/docker/daemon.json
{
    "registry-mirrors": [
		"https://docker.1panel.live",
		"https://image.cloudlayer.icu"
    ]
}

```
- These mirrors may be invalid, you need to search new ones.

```Bash
sudo systemctl daemon-reload
sudo systemctl restart docker.service

sudo docker pull sqreen/dvwa  # Try to pull an image.
```

# Execute Docker Images
```
docker exec -it 容器ID /bin/bash
```