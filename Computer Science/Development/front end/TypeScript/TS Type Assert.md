如果你确定`node.id`一定会返回字符串（不会是`undefined`），你可以使用类型断言（Type Assertion）来告诉编译器按照你的判断进行类型推断。以下是一种修改方法：
```typescript
const nodeId: string = node.id as string;
```

或者使用尖括号形式的类型断言：
```typescript
const nodeId: string = <string>node.id;
```

这样，你就告诉编译器将`node.id`视为一个字符串类型，消除了类型推断的错误。请确保在使用类型断言时，你能够确保`node.id`的值始终是字符串类型，否则可能会引发运行时错误。