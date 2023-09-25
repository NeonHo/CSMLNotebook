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
# 输出
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


F12能够通过Snippets中写一些小脚本，然后运行。

输出有两种方式：
通过修改HTML元素显示，
通过alert警告显示。

```HTML
<!DOCTYPE html>
<html>
<body>
<h1>我的第一个页面</h1><p>我的第一个段落。</p>
	
<script>window.alert(5 + 6);
</script>

</body>
</html>
```
或者通过`console.log()`写到控制台。

# 语法
## 字面量
一般的固定值
Number、String、表达式字面量、Array、Object、Function
## 变量
var 存储数据值。
