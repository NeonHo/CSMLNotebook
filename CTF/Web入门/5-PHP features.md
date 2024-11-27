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

Done! ![[Pasted image 20241127150837.png]]


## Experimentation
(Design and conduct experiments to test the hypothesis.)