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
sudo apt-get install docker.io
sudo apt-get install docker-compose


sudo docker -v
sudo docker-compose -v 
```

Docker change images:
```
vi /etc/docker/daemon.json
{
    "registry-mirrors": [
		"cr.laoyou.ip-ddns.com",
		"docker.1panel.live",
		"image.cloudlayer.icu",
    ]
}

```

# Execute Docker Images
```
docker exec -it 容器ID /bin/bash
```