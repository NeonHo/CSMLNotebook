# 请求方式
HTTP 请求方法, HTTP/1.1协议中共定义了八种方法（也叫动作）来以不同方式操作指定的资源。

其中Get可以这样写：
在Web应用的控制台上，假设你要在控制台上调用一个特定的GET方法，并在响应中获取flag，可以使用`fetch`方法并处理响应。

以下是一个示例代码：

```javascript
fetch('https://example.com/flag-endpoint', {
  method: 'GET',
})
.then(response => response.text())
.then(data => {
  console.log('Flag:', data);
})
.catch(error => console.error('Error:', error));
```

在这个示例中：
1. 将`https://example.com/flag-endpoint`替换为实际的URL。
2. 使用`.text()`或`.json()`等方法来处理响应数据。

### 示例解释
- `fetch` 方法发起一个GET请求。
- 使用`.then()`方法处理响应数据并在控制台中输出。

HTTP/1.1协议中定义了八种方法（动作），分别是GET、HEAD、POST、PUT、DELETE、CONNECT、OPTIONS和TRACE。以下是每种方法在控制台上使用`fetch` API执行的示例代码：
## HTTP 1.1 的 8种请求方式
### 1. GET
用于请求指定资源。

```javascript
fetch('https://example.com/resource', {
  method: 'GET'
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### 2. HEAD
类似于GET请求，但只请求响应的头部信息，不返回响应体。

```javascript
fetch('https://example.com/resource', {
  method: 'HEAD'
})
.then(response => {
  console.log('Headers:', response.headers);
})
.catch(error => console.error('Error:', error));
```

### 3. POST
用于提交数据到指定资源，通常用于表单提交。

```javascript
fetch('https://example.com/resource', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ key: 'value' })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### 4. PUT
用于更新指定资源。

```javascript
fetch('https://example.com/resource/1', {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ key: 'updated value' })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### 5. DELETE
用于删除指定资源。

```javascript
fetch('https://example.com/resource/1', {
  method: 'DELETE'
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### 6. CONNECT
用于建立隧道连接以进行通信。此方法通常在客户端不直接使用，更多在代理服务器中使用。

```javascript
fetch('https://example.com/resource', {
  method: 'CONNECT'
})
.then(response => console.log('Connected'))
.catch(error => console.error('Error:', error));
```

### 7. OPTIONS
用于获取当前URL所支持的方法。

```javascript
fetch('https://example.com/resource', {
  method: 'OPTIONS'
})
.then(response => console.log(response.headers.get('Allow')))
.catch(error => console.error('Error:', error));
```

### 8. TRACE
用于对请求进行回显，主要用于诊断。

```javascript
fetch('https://example.com/resource', {
  method: 'TRACE'
})
.then(response => response.text())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

请注意：
- `CONNECT`方法在大多数浏览器的控制台中不常用，更多用于代理服务器。
- 使用这些示例代码时，请确保URL是实际可用的，并根据实际需求调整请求的内容和处理方式。

## curl工具

**curl 命令需要在控制台上进行执行，Safari浏览器不能用。**

`curl` 是一个命令行工具，用于通过各种协议（如HTTP、HTTPS、FTP等）传输数据。下面是对这条命令 `curl -v -X GET https://example.com/resource` 的解释：

### 解释命令

- `curl`：调用curl工具。
- `-v`：启用详细模式（verbose），这将使curl显示详细的请求和响应信息，包括头部信息、响应码等。这对于调试和了解请求的详细情况非常有用。
- `-X GET`：指定HTTP方法为GET。虽然GET是默认方法，但明确指定可以确保使用GET方法进行请求。
- `https://example.com/resource`：目标URL，表示要向这个URL发起GET请求。

### 等效的JavaScript代码

在浏览器控制台中，等效于上述`curl`命令的JavaScript代码如下：

```javascript
fetch('https://example.com/resource', {
  method: 'GET'
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

如果你想在控制台中看到详细的请求和响应信息，可以使用浏览器的开发者工具来查看网络活动。在大多数浏览器中，可以通过以下步骤实现：

1. 打开开发者工具（通常可以通过按下 `F12` 或 `Ctrl+Shift+I` 打开）。
2. 选择“网络”或“Network”标签。
3. 执行JavaScript代码后，在“网络”标签中可以看到所有的网络请求和详细的请求/响应信息。

### 更详细的解释

#### `curl`
`curl`是一个命令行工具，用于向服务器发送请求并获取数据。

#### `-v`
`-v`选项启用详细模式，使curl显示发送的请求和接收到的响应的详细信息。这包括请求头、响应头和其他调试信息。

#### `-X GET`
`-X`选项用于指定HTTP方法。虽然GET是默认方法，但明确指定可以确保使用GET方法。

#### `https://example.com/resource`
这是目标URL，表示要向这个URL发起GET请求。

### 总结
这条`curl`命令用于向指定URL（`https://example.com/resource`）发起GET请求，并以详细模式显示请求和响应的全部信息。这对于调试和分析请求非常有帮助。

# 临时重定向
HTTP临时重定向是指服务器向客户端（如浏览器）发送一个状态码，指示客户端应临时访问一个不同的URL，而不是原始请求的URL。临时重定向的常见状态码是302和307。以下是对HTTP临时重定向的详细解释：

### 302 Found（临时重定向）

#### 含义
- 302状态码表示所请求的资源临时地被移动到另一个URL，客户端应继续使用原始请求的URL来进行后续请求。
- 服务器在响应中包含`Location`头部，指示客户端应访问的新URL。

#### 示例
假设你请求`http://example.com/old-page`，服务器响应如下：

```
HTTP/1.1 302 Found
Location: http://example.com/new-page
```

浏览器会自动重定向到`http://example.com/new-page`，但用户在将来仍然可以使用`http://example.com/old-page`来访问原始资源。

#### 示例代码（使用curl）
```sh
curl -v -X GET http://example.com/old-page
```

### 307 Temporary Redirect（临时重定向）

#### 含义
- 307状态码类似于302，但更严格地要求客户端在重定向请求时保持原始请求方法和请求体。
- 服务器在响应中包含`Location`头部，指示客户端应访问的新URL。

#### 示例
假设你请求`http://example.com/old-page`，服务器响应如下：

```
HTTP/1.1 307 Temporary Redirect
Location: http://example.com/new-page
```

浏览器会自动重定向到`http://example.com/new-page`，并使用与原始请求相同的方法（例如GET或POST）。

#### 示例代码（使用curl）
```sh
curl -v -X GET http://example.com/old-page
```

### 使用场景
- **302 Found**：多用于临时重定向，比如当某个资源暂时不可用而需要转移到另一个URL时。浏览器可能会缓存302响应的重定向目标。
- **307 Temporary Redirect**：更常用于需要确保请求方法不变的场景，例如表单提交的重定向。

### 浏览器行为
当浏览器收到302或307状态码时，会自动处理重定向并访问新的URL。这对用户是透明的，用户不会注意到中间的重定向过程。

### 示例代码（使用JavaScript的fetch）
假设你在浏览器控制台中发送一个请求并处理重定向响应：

```javascript
fetch('http://example.com/old-page')
  .then(response => {
    if (response.status === 302 || response.status === 307) {
      return fetch(response.headers.get('Location'));
    } else {
      return response;
    }
  })
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
```

### 总结
HTTP临时重定向允许服务器暂时指示客户端访问一个新的URL而不改变原始请求的URL。302和307是两种常见的临时重定向状态码，它们的主要区别在于是否要求重定向请求保持原始方法和请求体。


# Cookie欺骗、认证和伪造

### 1. Cookie基础知识
**Cookie** 是服务器在客户端（如浏览器）上存储的一小段数据，用于保持会话状态、存储用户偏好、跟踪用户活动等。Cookies包含一个名称-值对，可以设置为在特定时间段内有效。

### 2. Cookie欺骗
**Cookie欺骗（Cookie Tampering）** 是指恶意用户修改或伪造Cookie，以便冒充合法用户或获取未授权的访问权限。

#### 示例：
- 攻击者通过拦截并修改Cookie中的值来更改用户身份或权限。例如，将用户ID从`123`改为`456`，以访问其他用户的账户。

### 3. Cookie认证
**Cookie认证** 是指使用Cookie来验证用户身份。服务器在用户登录时生成一个会话ID（Session ID）并将其存储在Cookie中。每次请求时，浏览器会自动发送该Cookie，以证明用户的身份。

#### 过程：
1. 用户登录时，服务器生成一个唯一的会话ID，并将其存储在Cookie中。
2. 客户端在后续请求中自动发送该Cookie，服务器通过检查Cookie中的会话ID来验证用户身份。

### 4. Cookie伪造
**Cookie伪造（Cookie Forging）** 是指攻击者创建或伪造合法的Cookie，以便冒充合法用户。这通常涉及对Cookie结构和加密机制的深入了解。

#### 示例：
- 攻击者使用工具或脚本创建一个伪造的会话Cookie，并注入到浏览器中，冒充合法用户访问受保护的资源。

### 如何防范Cookie欺骗和伪造

#### 1. 使用HttpOnly标志
将Cookie标记为`HttpOnly`，使其无法通过JavaScript访问，减少XSS攻击带来的风险。

```http
Set-Cookie: sessionId=abc123; HttpOnly
```

#### 2. 使用Secure标志
将Cookie标记为`Secure`，确保它只能通过HTTPS连接传输，防止在传输过程中被拦截。

```http
Set-Cookie: sessionId=abc123; Secure
```

#### 3. 设置SameSite标志
使用`SameSite`标志，限制Cookie在跨站请求中的发送，从而防止CSRF攻击。

```http
Set-Cookie: sessionId=abc123; SameSite=Strict
```

#### 4. 加密和签名
对Cookie中的敏感数据进行加密和签名，防止其被篡改。

```javascript
// 示例：加密和签名Cookie
const crypto = require('crypto');

function createSignedCookie(value, secret) {
  const signature = crypto.createHmac('sha256', secret).update(value).digest('hex');
  return `${value}.${signature}`;
}

const secret = 'mySecretKey';
const cookieValue = 'userId=123';
const signedCookie = createSignedCookie(cookieValue, secret);
console.log(signedCookie);
```

#### 5. 定期验证
服务器应定期验证Cookie中的数据，确保其未被篡改。例如，通过检查Cookie中的签名或加密值来验证其完整性。

### 总结
- **Cookie欺骗**：通过修改Cookie值来冒充其他用户或提升权限。
- **Cookie认证**：使用Cookie存储会话ID以验证用户身份。
- **Cookie伪造**：创建或伪造合法的Cookie以冒充合法用户。
- **防范措施**：使用HttpOnly、Secure和SameSite标志，实施加密和签名，定期验证Cookie数据。
# 基本认证
https://zh.wikipedia.org/wiki/HTTP基本认证

在[HTTP](https://zh.wikipedia.org/wiki/HTTP "HTTP")中，**基本认证**（英语：Basic access authentication）是允许[http用户代理](https://zh.wikipedia.org/wiki/%E7%94%A8%E6%88%B7%E4%BB%A3%E7%90%86 "用户代理")（如：[网页浏览器](https://zh.wikipedia.org/wiki/%E7%BD%91%E9%A1%B5%E6%B5%8F%E8%A7%88%E5%99%A8 "网页浏览器")）在请求时，提供 [用户名](https://zh.wikipedia.org/wiki/%E7%94%A8%E6%88%B7%E5%90%8D "用户名") 和 [密码](https://zh.wikipedia.org/wiki/%E5%8F%A3%E4%BB%A4 "密码") 的一种方式。

在进行基本认证的过程里，请求的[HTTP头字段](https://zh.wikipedia.org/wiki/HTTP%E5%A4%B4%E5%AD%97%E6%AE%B5 "HTTP头字段")会包含`Authorization`字段，形式如下： `Authorization: Basic <凭证>`，该凭证是用户和密码的组和的[base64编码](https://zh.wikipedia.org/wiki/Base64 "Base64")。

最初，基本认证是定义在HTTP 1.0规范（[RFC 1945](https://tools.ietf.org/html/rfc1945)）中，后续的有关安全的信息可以在HTTP 1.1规范（[RFC 2616](https://tools.ietf.org/html/rfc2616)）和HTTP认证规范（[RFC 2617](https://tools.ietf.org/html/rfc2617)）中找到。于1999年 [RFC 2617](https://tools.ietf.org/html/rfc2617) 过期，于2015年的 [RFC 7617](https://tools.ietf.org/html/rfc7617) 重新被定义。

在[MDN](https://zh.wikipedia.org/wiki/MDN_Web_Docs "MDN Web Docs")网站，已经有对应的维基文章[1](https://zh.wikipedia.org/wiki/HTTP%E5%9F%BA%E6%9C%AC%E8%AE%A4%E8%AF%81#cite_note-1)。

## 具体题目操作
首先题目给出了前100个密码，但是没有告诉我们用户名，所以我们需要使用intruder暴力破解。
![[Pasted image 20240713122324.png]]
首先我们随便用一个编造名和一个随意的密码试一下：
```
GET /flag.html HTTP/1.1
Host: challenge-8bfcd162ce5be763.sandbox.ctfhub.com:10800
Cache-Control: max-age=0
Authorization: Basic Y3RmOjEyMzQ1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Referer: http://challenge-8bfcd162ce5be763.sandbox.ctfhub.com:10800/
Accept-Encoding: gzip, deflate
Accept-Language: zh-CN,zh;q=0.9
Connection: close


```
其中Authorization后面Basic标示Base编码，`Y3RmOjEyMzQ1`就是Base加密的，我们可以将其Convert回来：
![[Pasted image 20240713122934.png]]
解码得到的账户和密码就是我们编的：`ctf:12345`
所以，我们只需要暴力搜这个Basic后的编码就可以。
这就需要使用intruder来进行attack。
具体需要设置四个位置：

**第一个是最开始的Host和端口号**
![[Pasted image 20240713122324.png]]
第二个是Position，把GET Message复制进去，并且add position: