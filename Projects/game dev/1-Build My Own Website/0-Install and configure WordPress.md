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
