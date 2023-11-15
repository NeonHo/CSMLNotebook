# `let`
在 JavaScript 中，`let` 是用于声明变量的关键字。`let` 关键字引入了块作用域的概念，即变量的作用范围限定在它所在的块（通常是花括号 `{}`）内。

```javascript
// 使用 let 声明变量
let myVariable = 10;

// 在块级作用域内使用 let
if (true) {
  let blockVariable = 20;
  console.log(blockVariable); // 输出 20
}

// 在块级作用域外访问 let 变量会导致 ReferenceError
// console.log(blockVariable); // 报错：blockVariable is not defined

// 可以在同一作用域内重新声明 let 变量，但不可以重新声明已存在的变量
let myVariable = 30; // 报错：Identifier 'myVariable' has already been declared
```

`let` 的特性有助于避免变量提升（hoisting）和解决在使用 `var` 声明时可能引发的一些问题。因为 `let` 在块级作用域内生效，所以它更符合直觉，并提供更好的代码组织和维护性。