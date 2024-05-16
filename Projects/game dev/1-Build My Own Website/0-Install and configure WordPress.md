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

[[Apache2]]
[[Ghostscript]]
# MySQL -> WordPress
1. [[Create Database for wordpress]]
2. configure WordPress to use this database.
3. copy the sample configuration file to `wp-config.php`
	1. `sudo cp softwares/wordpress/wp-config-sample.php softwares/wordpress/wp-config.php`
4. set the database credentials in the configuration file
	1. `sudo sed -i 's/database_name_here/wordpress/' softwares/wordpress/wp-config.php`
	2. `sudo sed -i 's/username_here/wordpress/' softwares/wordpress/wp-config.php`
	3. 
