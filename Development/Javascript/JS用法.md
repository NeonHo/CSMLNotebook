JS脚本一定是位于H5的 `<script>` 与 `</script>` 标签之间。
而 这个包裹必须在`<body>`和`<head>` .

可以在加载时执行，
可以在调用函数时执行
可以在某个事件发生时发生。

```JavaScript
x=document.getElementById("demo");  //查找元素
x.innerHTML="Hello JavaScript";    //改变内容
```

根据ID获得一个DOM，对这个DOM进行修改。

完整的内容可以简单的理解为：
```HTML
<!DOCTYPE html>
<html>
<body>
<h1>我的 Web 页面</h1>
<p id="demo">一个段落</p>
<button type="button" onclick="myFunction()">尝试一下</button>
<script>
function myFunction()
{
    document.getElementById("demo").innerHTML="我的第一个 JavaScript 函数";
}
</script>
</body>
</html>
```


F12能够通过Snippets中写一些小脚本，然后运行
