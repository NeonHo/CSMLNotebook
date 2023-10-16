# `ref()`
在 Vue.js 中，`ref` 是用于在组件中创建响应式数据的函数。

如果你想让你的变量能够显示在`<template>`中，那就一定要使用这个作为变量的赋值。
## `const` & `let`
`const` 修饰符是 JavaScript 中的一种声明方式，用于创建不可重新分配的常量。将 `const` 修饰符与 `ref` 一起使用有一些特定的含义和用途。

当你在 Vue.js 组件中使用 `ref` 创建一个变量时，通常你会使用 `let` 或 `const` 来声明这个变量。`const` 的主要特点是一旦分配了值，就不能再重新分配新的值给这个变量。这意味着该变量在其生命周期内将保持不变。

在某些情况下，将 `const` 与 `ref` 一起使用是有意义的，特别是当你希望确保这个变量在组件中不被重新分配。这可以有助于提高代码的可维护性和可读性，因为开发人员可以清晰地知道这个变量不会被意外地修改。

以下是一个示例，演示如何在 Vue.js 组件中使用 `const` 和 `ref`：

```vue
<template>
  <div>
    <p>Count: {{ count }}</p>
    <button @click="increment">Increment</button>
  </div>
</template>

<script>
import { ref } from 'vue';

export default {
  setup() {
    // 使用 const 声明一个不可重新分配的常量
    const count = ref(0);

    // 增加计数的函数
    function increment() {
      count.value++;
    }

    return {
      count,
      increment
    };
  }
};
</script>
```

在这个示例中，`count` 被声明为一个 `const` 常量，并使用 `ref` 创建为一个响应式变量。这确保了 `count` 在组件内部不会被重新分配，但仍然可以使用 `count.value` 来访问其响应式值并修改它。
# 根据其他component刷新某个component的思路

设置一个app.vue的ref变量。
```Vue.js
const graphUpdateTrigger = ref(1);
```
在要刷新的组件上做：
```Vue.js
var graphUpdateTrigger = inject('graphUpdateTrigger')
```
通过`watch`方法看`graphUpdateTrigger` 变量的改变，然后触发后端请求，从而更新UI。

# vue3中的
# 注意
前端的post和get方法，一定要和后端是匹配的，不然会在Axios调用时报错。
