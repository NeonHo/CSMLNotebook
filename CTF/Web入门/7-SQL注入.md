# Web 171
![[Pasted image 20241214102338.png]]
```PHP
//拼接sql语句查找指定ID用户
$sql = "select username,password from user where username !='flag' and id = '".$_GET['id']."' limit 1;";
```

```SQL
select username,password from user where username !='flag' and id = '".$_GET['id']."' limit 1;
```
## Observation
PHP code is : `$sql = "xxxxx".$_GET['id']."yyyy"`
- `xxxxx`: `select username,password from user where username !='flag' and id = '`
- `yyyy`: `' limit 1; `
When we do search in this page:
- If username is not flag, and the user ID is not what we provide, the records will be displayed.
- Limit will be executed at last.
## Research

This is a typical string **SQL Injection**.
- 