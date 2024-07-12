# 1. 请求方式
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
