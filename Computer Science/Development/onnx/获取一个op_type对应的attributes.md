你可以使用以下方法来探索 ONNX 运算符及其属性：

1. **ONNX 官方文档：** 在 [ONNX GitHub 仓库](https://github.com/onnx/onnx) 的 `docs/Operators.md` 文件中，你可以找到每个运算符的详细描述，包括支持的属性。

2. **onnx 模块的文档：** 在 ONNX Python 包的文档中，你可以查看 `onnx.NodeProto` 类和其他相关类的文档。这些类包含了表示 ONNX 运算符节点的信息。

3. **源代码：** 如果你对 ONNX 的实现感兴趣，可以查看 ONNX 的源代码。ONNX 的 GitHub 仓库包含了所有运算符的定义，你可以在 `onnx/onnx/backend/test/data/` 目录下找到相关的测试数据，它们以 `.onnx` 格式存储。

4. **使用 Python 代码获取：** 你可以编写一个脚本来自动检索 ONNX 运算符的信息。以下是一个简单的示例，该示例使用 ONNX 包中的 `get_all_predefined_operators()` 函数获取所有预定义的运算符，然后打印每个运算符及其属性：

```python
import onnx

def get_all_attributes():
    operators = onnx.defs.get_all_predefined_operators()
    for op in operators:
        print(f"\nOperator: {op}")
        schema = onnx.defs.get_schema(op)
        for attr in schema.attributes:
            print(f"Attribute: {attr.name}, Type: {attr.type}, Default Value: {attr.default_value}")

if __name__ == "__main__":
    get_all_attributes()
```

请注意，这个示例仅显示了 ONNX 运算符的名称、属性名称、类型和默认值。你可能还需要查阅 ONNX 官方文档来获得更详细的信息。