# 1. 请求方式
HTTP 1.1 有8种请求方式

其中Get可以这样写：
在Web应用的控制台上，假设你要在控制台上调用一个特定的GET方法，并在响应中获取flag，可以使用`fetch`方法并处理响应。

以下是一个示例代码：

```javascript
fetch('https://example.com/flag-endpoint', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer YOUR_FLAG'  // 假设需要提供某种认证信息
  }
})
.then(response => response.text())
.then(data => {
  console.log('Flag:', data);
})
.catch(error => console.error('Error:', error));
```

在这个示例中：
1. 将`https://example.com/flag-endpoint`替换为实际的URL。
2. 将`YOUR_FLAG`替换为实际的flag或者认证信息，如果不需要认证则可以省略`headers`部分。
3. 使用`.text()`或`.json()`等方法来处理响应数据。

### 示例解释
- `fetch` 方法发起一个GET请求。
- `headers` 部分可以根据需要添加，比如认证信息。
- 使用`.then()`方法处理响应数据并在控制台中输出。

如果具体的`CTF`方法涉及更加复杂的逻辑，请提供更多的细节以便提供更精准的示例。