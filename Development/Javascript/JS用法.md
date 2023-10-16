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
等号赋值。
## 操作符
算术运算符、赋值运算符。
## 语句
用分号隔开
## 关键字
var创建一个变量。
## 数据类型
数字、表达式、字符串、数组、对象。
## 函数
```JavaScript
function myFunction(a, b) {
   	return a * b;         
                     
// 返回 a 乘以 b 的结果
}
```
# 变量生存期
JavaScript 变量的生存期
JavaScript 变量的生命期从它们被声明的时间开始。

局部变量会在函数运行以后被删除。

全局变量会在页面关闭后被删除。

# 未声明类型的变量赋值会怎么样？

如果您把值赋给尚未声明的变量，该变量将被自动作为 window 的一个属性。

这条语句：

carname="Volvo";
将声明 window 的一个属性 carname。

非严格模式下给未声明变量赋值创建的全局变量，是全局对象的可配置属性，可以删除。

# 事件
HTML事件 浏览器行为、用户行为。
当事件发生， 就可以用JS执行代码。
```JavaScript
<some-HTML-element some-event='JavaScript 代码'>
```
例如，点击按钮显示时间
```JavaScript
<button onclick="getElementById('demo').innerHTML=Date()">现在的时间是?</button>
<p id="demo"></p>
```
# 变量提升
JavaScript 中，函数及变量的声明都将被提升到函数的最顶部。

JavaScript 中，变量可以在使用后声明，也就是变量可以先使用再声明。
（因为变量提升，所以声明在哪里都不受影响。）
# this
在方法中，this 表示该方法所属的对象。
如果单独使用，this 表示全局对象。
在函数中，this 表示全局对象。
在函数中，在严格模式下，this 是未定义的(undefined)。
在事件中，this 表示接收事件的元素。
类似 call() 和 apply() 方法可以将 this 引用到任何对象。
# let & const

let 声明的变量只在 let 命令所在的代码块内有效。

const 声明一个只读的常量，一旦声明，常量的值就不能改变。

# 异步
JavaScript 中的异步操作函数往往通过回调函数来实现异步任务的结果处理。
```JavaScript
<p>回调函数等待 3 秒后执行。</p>
<p id="demo1"></p>
<p id="demo2"></p>
<script>
setTimeout(function () {
    document.getElementById("demo1").innerHTML="RUNOOB-1!";  // 三秒后子线程执行
}, 3000);
document.getElementById("demo2").innerHTML="RUNOOB-2!";      // 主线程先执行
</script>

```
RUNOOB-1!会在3秒后才显示出来，但是RUNOOB-2!立刻显示。

