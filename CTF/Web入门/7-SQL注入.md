# Web 171
![[Pasted image 20241214102338.png]]
```PHP
//拼接sql语句查找指定ID用户
$sql = "select username,password from user where username !='flag' and id = '".$_GET['id']."' limit 1;";
```

```
```