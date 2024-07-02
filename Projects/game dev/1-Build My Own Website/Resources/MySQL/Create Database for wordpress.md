[参考 step1](https://ubuntu.com/tutorials/install-and-configure-wordpress#5-configure-database)
[参考 step2](https://ubuntu.com/tutorials/install-and-configure-wordpress#6-configure-wordpress-to-connect-to-the-database)
这一套操作在 MySQL 中完成了以下任务：

1. **创建数据库 `wordpress`**：
   ```sql
   CREATE DATABASE wordpress;
   ```
   这条命令创建了一个名为 `wordpress` 的新数据库，将用于存储 WordPress 的所有数据。

2. **创建用户 `wordpress`@`localhost` 并设置密码**：
   ```sql
   CREATE USER wordpress@localhost IDENTIFIED BY '<your-password>';
   ```
   这条命令创建了一个名为 `wordpress` 的新用户，该用户只能从 `localhost` 登录，并设置了一个密码（用 `<your-password>` 替换实际密码）。`localhost` 表示这个用户只能从本地主机访问数据库。

3. **授予权限**：
   ```sql
   GRANT SELECT,INSERT,UPDATE,DELETE,CREATE,DROP,ALTER
   ON wordpress.*
   TO wordpress@localhost;
   ```
   这条命令授予 `wordpress` 用户对 `wordpress` 数据库的特定权限：
   - `SELECT`：允许读取数据。
   - `INSERT`：允许插入数据。
   - `UPDATE`：允许更新数据。
   - `DELETE`：允许删除数据。
   - `CREATE`：允许创建新的表和数据库对象。
   - `DROP`：允许删除表和数据库对象。
   - `ALTER`：允许修改现有的表结构。

   这些权限确保 `wordpress` 用户有权执行 WordPress 所需的所有数据库操作。

4. **刷新权限**：
   ```sql
   FLUSH PRIVILEGES;
   ```
   这条命令使刚才的权限更改立即生效。MySQL 有时会缓存用户和权限信息，`FLUSH PRIVILEGES` 命令强制 MySQL 重新加载这些信息。

5. **退出 MySQL**：
   ```sql
   quit
   ```
   这条命令退出 MySQL 命令行客户端。

### 总结

通过这套操作，你已经成功完成了以下任务：
- 创建了一个用于存储 WordPress 数据的新数据库 `wordpress`。
- 创建了一个新用户 `wordpress` 并设置了密码。
- 授予了该用户对 `wordpress` 数据库的必要权限。
- 刷新了 MySQL 的权限设置，使新的权限立即生效。

这些步骤确保了 WordPress 应用程序可以使用 `wordpress` 用户来连接和操作 `wordpress` 数据库，从而正常运行。