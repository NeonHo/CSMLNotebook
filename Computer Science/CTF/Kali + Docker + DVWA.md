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
```
DVWA 是一个入门的 Web 安全学习靶场，说简单也不简单，结合源码去学习的话，不仅可以入门安全也还可以学到不少安全加固的知识，个人认为国光我写的这个在 DVWA 靶场教程中算是比较细致全面的了。
```Bash
sudo docker pull sqreen/dvwa  # Try to pull an image.
```
# Run Docker Image
```Bash
docker run -d -t -p 8888:80 sqreen/dvwa
```
# Execute Docker Images
```
docker exec -it 容器ID /bin/bash
```