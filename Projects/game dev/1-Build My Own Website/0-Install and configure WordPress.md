[Medium Ref](https://medium.com/@sabinjoshi044/how-to-install-wordpress-on-ubuntu-a-step-by-step-guide-bad460da8fa5)

# Install dependencies
[tutorial](https://ubuntu.com/tutorials/install-and-configure-wordpress#2-install-dependencies)
```Bash
sudo apt update
sudo apt install apache2 \
                 ghostscript \
                 libapache2-mod-php \
                 mysql-server \
                 php \
                 php-bcmath \
                 php-curl \
                 php-imagick \
                 php-intl \
                 php-json \
                 php-mbstring \
                 php-mysql \
                 php-xml \
                 php-zip
```

[[Ghostscript]]
## Install Apache2
[[Apache2]]

Apache2的详细安装和指令
[参考2](http://t.csdnimg.cn/NsxqT)

阿里云中需要对安全组进行编辑
[参考1](http://t.csdnimg.cn/GsnIq)
![[Pasted image 20240516174121.png]]

不然访问不了Apache的 “It works!”页面
## Install PHP
```bash
sudo apt install -y php php-{common,mysql,xml,xmlrpc,curl,gd,imagick,cli,dev,imap,mbstring,opcache,soap,zip,intl}
```
http://t.csdnimg.cn/bWf1w
# MySQL -> WordPress
1. [[Create Database for wordpress]]
2. configure WordPress to use this database.
3. copy the sample configuration file to `wp-config.php`
	1. `sudo cp softwares/wordpress/wp-config-sample.php softwares/wordpress/wp-config.php`
4. set the database credentials in the configuration file
	1. `sudo sed -i 's/database_name_here/wordpress/' softwares/wordpress/wp-config.php`
	2. `sudo sed -i 's/username_here/wordpress/' softwares/wordpress/wp-config.php`
	3. 
