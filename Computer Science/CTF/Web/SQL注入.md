SQL注入漏洞是指攻击者通过浏览器或者其他客户端将恶意SQL语句插入到网站参数中，而网站应用程序未对其进行过滤，将恶意SQL语句带入数据库使恶意SQL语句得以执行，从而使攻击者通过数据库获取敏感信息或者执行其他恶意操作。
- 绕过登录验证：使用万能密码登录网站后台等。 如：'or 1=1#
- 获取敏感数据：获取网站管理员帐号、密码等。
- 文件系统操作：列目录，读取、写入文件等。
- 执行系统命令：远程执行命令。

`--dbs` get the databases.
`-D` specify the database, following:
- `--tables` get the tables in this database.
- `-T` specify the table, following:
	- `--columns` shows the columns.
	- `-C` specify the columns, following:
		- `--dump` print the values for each columns.

