`cURL`（Client for URLs）是一个强大的**命令行工具**和**库**（libcurl），用于通过各类网络协议（如HTTP、HTTPS、FTP等）传输数据。它支持多种功能，包括发送请求、下载文件、测试API等，是开发者、运维和安全测试人员的常用工具。

---

## **1. cURL的核心功能**
### **(1) 发送HTTP请求**
- **GET请求**（默认）：
  ```bash
  curl https://example.com
  ```
- **POST请求**（提交表单或JSON）：
  ```bash
  curl -X POST -d 'name=Alice&age=25' https://example.com/api
  curl -X POST -H "Content-Type: application/json" -d '{"name":"Alice"}' https://example.com/api
  ```

### **(2) 下载/上传文件**
- **下载文件**：
  ```bash
  curl -O https://example.com/file.zip  # 保存为原文件名
  curl -o custom_name.zip https://example.com/file.zip  # 自定义文件名
  ```
- **上传文件**：
  ```bash
  curl -F "file=@localfile.txt" https://example.com/upload
  ```

### **(3) 调试与测试**
- **显示请求头**：
  ```bash
  curl -I https://example.com  # 仅获取响应头
  curl -v https://example.com  # 显示详细请求/响应过程
  ```
- **模拟浏览器**（携带User-Agent和Cookie）：
  ```bash
  curl -A "Mozilla/5.0" -b "sessionid=123" https://example.com
  ```

### **(4) 支持多种协议**
- **HTTP/HTTPS**、**FTP**、**SFTP**、**SCP**、**SMTP**等：
  ```bash
  curl ftp://example.com/file.txt
  curl smtp://mail.example.com --mail-from sender@example.com --mail-rcpt receiver@example.com
  ```

---

## **2. 常见使用场景**
### **(1) API测试与调试**
- **快速验证接口**：
  ```bash
  curl -X GET "https://api.example.com/users?id=1"
  ```
- **认证请求**（Bearer Token/Basic Auth）：
  ```bash
  curl -H "Authorization: Bearer token123" https://api.example.com
  curl -u username:password https://api.example.com
  ```

### **(2) 自动化脚本**
- **定时获取数据**：
  ```bash
  # 每天下载日志
  curl -o /var/log/daily.log https://example.com/logs
  ```
- **与`jq`结合处理JSON响应**：
  ```bash
  curl -s https://api.example.com/data | jq '.results[0].name'
  ```

### **(3) 网络问题排查**
- **检查HTTP状态码**：
  ```bash
  curl -s -o /dev/null -w "%{http_code}" https://example.com
  ```
- **测试重定向**：
  ```bash
  curl -L https://example.com  # 自动跟随重定向
  ```

### **(4) 安全测试**
- **SSRF探测内网服务**：
  ```bash
  curl http://127.0.0.1:8080/admin
  ```
- **文件包含漏洞利用**：
  ```bash
  curl "http://victim.com/?file=../../etc/passwd"
  ```

### **(5) 数据备份与同步**
- **镜像网站**：
  ```bash
  curl -o index.html https://example.com
  ```
- **FTP批量下载**：
  ```bash
  curl -u user:pass -O "ftp://example.com/files/*.txt"
  ```

---

## **3. 高级用法示例**
### **(1) 限速下载**
```bash
curl --limit-rate 100K -O https://example.com/largefile.iso
```

### **(2) 断点续传**
```bash
curl -C - -O https://example.com/bigfile.zip
```

### **(3) 代理请求**
```bash
curl -x http://proxy.example.com:8080 https://target.com
```

### **(4) 保存Cookie并复用**
```bash
curl -c cookies.txt https://example.com/login
curl -b cookies.txt https://example.com/dashboard
```

---

## **4. 与`wget`的区别**
| **特性**          | **cURL**                          | **wget**                        |
|--------------------|-----------------------------------|---------------------------------|
| **协议支持**       | 更多（如SMTP、SCP）               | 主要HTTP/HTTPS/FTP              |
| **脚本友好度**     | 更适合API交互和管道操作           | 更适合下载文件                  |
| **递归下载**       | 不支持                            | 支持（`-r`参数）                |
| **默认行为**       | 输出到stdout                      | 直接下载到文件                  |

---

## **5. 注意事项**
1. **敏感信息泄露**：  
   避免在命令行中直接暴露密码或Token（可用`-n`从`.netrc`读取）。
2. **HTTPS证书验证**：  
   测试时可跳过证书验证（`-k`参数），但生产环境禁用。
3. **用户代理伪装**：  
   某些网站屏蔽`curl`的默认User-Agent，需模拟浏览器：
   ```bash
   curl -A "Mozilla/5.0" https://example.com
   ```

---

## **总结**
- **何时使用cURL**：  
  需要快速发送请求、测试API、下载文件、调试网络或自动化脚本时。
- **核心优势**：  
  支持多种协议、灵活的参数配置、易于集成到脚本中。
- **经典命令**：  
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' https://api.example.com
  ```

掌握cURL能显著提升开发、运维和安全测试的效率，是命令行工具链中的瑞士军刀。