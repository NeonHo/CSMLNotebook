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
In the standard SQL, there is no base64 encoding method.
However, 
## Experiment
