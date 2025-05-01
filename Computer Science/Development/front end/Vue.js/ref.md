在 Vue 3 中，`ref` 是用于创建响应式对象的函数。它的主要功能是将普通的 JavaScript 数据变成响应式对象，使其能够在 Vue 组件中进行双向绑定以及在模板中实时反映数据的变化。

具体来说，`ref` 接受一个参数，该参数可以是基本类型（比如数字、字符串、布尔值等）或对象。`ref` 返回一个包装过的响应式对象，这个对象有一个 `.value` 属性，该属性指向传入的参数。当你修改 `.value` 属性时，Vue 将会检测到这个变化并触发相关的重新渲染。

以下是一个简单的示例：

```javascript
import { ref } from 'vue';

// 创建一个 ref 对象
const count = ref(0);

// 在模板中使用 ref 对象
console.log(count.value); // 输出当前值

// 修改 ref 对象的值
count.value += 1; // 触发重新渲染
```

在这个例子中，`count` 是一个 ref 对象，初始值为 0。在模板中，我们通过 `count.value` 来访问其当前值，通过修改 `count.value` 来更新值。这样做的好处是，Vue 能够追踪对 `count.value` 的修改，并在值变化时自动更新视图。