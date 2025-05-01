要获取 ONNX 中所有支持的运算符类型 (`op_type`)，你可以使用 ONNX 的 Python API。以下是一个简单的示例：

```python
import onnx

# 获取支持的运算符集合
all_ops = onnx.defs.get_all_operator_schema()
supported_op_types = set(op.name for op in all_ops)

# 打印支持的运算符类型
for op_type in sorted(supported_op_types):
    print(op_type)
```

在这个例子中，`onnx.defs.get_all_operator_schema()` 返回一个包含所有支持运算符的列表。然后，我们从中提取每个运算符的名称 (`name`) 并将其打印出来。

请注意，ONNX 版本可能会影响支持的运算符，所以确保你使用的是最新版本的 ONNX。