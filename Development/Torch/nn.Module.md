# register_forward_hook & register_backward_hook
在 PyTorch 中，`nn.Module` 类的 `register_forward_hook` 和 `register_backward_hook` 方法可以用来注册前向和后向钩子函数（hook function），以便在模型的前向传递和反向传递过程中执行一些额外的操作。<br />`register_forward_hook` 方法接受一个函数作为参数，该函数将在模型执行前向传递时被调用，并接收三个参数：模型实例、输入张量和输出张量。这允许您查看模型的输入和输出，并在需要时对其进行修改或记录。例如，您可以使用前向钩子函数来记录模型的中间输出，以便进行可视化或调试：
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def my_hook(module, input, output):
    print("Module:", module)
    print("Input shape:", input[0].shape)
    print("Output shape:", output.shape)

model = MyModel()
model.register_forward_hook(my_hook)

x = torch.randn(1, 3, 32, 32)
y = model(x)
```
如果要在这个`hook`函数能把东西给到某个对象, 就在这个对象类`C`中新建一个带闭包的成员函数.
```python
class C:
    def __init__(self):
        self.inputs = {"name1": None}
        self.outputs = {"name2": None}
        
    def generate_hook(self, name, module):
        def f(module, input, output):
            self.inputs[name] = input
            self.outputs[name] = output
        return f
```

`register_backward_hook` 方法接受一个函数作为参数，该函数将在模型执行反向传递时被调用，并接收四个参数：模型实例、梯度张量的元组、输入张量的元组和输出张量。这允许您查看模型的梯度，并在需要时对其进行修改或记录。例如，您可以使用后向钩子函数来记录梯度的分布情况，以便进行调试：
```python
def my_hook(module, grad_input, grad_output):
    print("Module:", module)
    print("Gradient input shape:", [gi.shape for gi in grad_input])
    print("Gradient output shape:", grad_output[0].shape)

model = MyModel()
model.register_backward_hook(my_hook)

x = torch.randn(1, 3, 32, 32)
y = model(x)
loss = y.sum()
loss.backward()
```

需要注意的是，钩子函数不应该修改模型的输入或输出，也不应该修改梯度张量。如果需要修改模型的输入或输出，应该使用 `nn.Module` 子类的 `forward` 方法。如果需要修改梯度张量，应该使用 `grad_input `或 `grad_output` 中的张量进行操作，而不能直接修改它们。
# register_buffer & register_parameter
该方法的作用是定义一组参数，该组参数的特别之处在于：模型训练时不会更新（即调用 `optimizer.step()` 后该组参数不会变化，只可人为地改变它们的值），但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。<br />modules和parameters可以被更新，而buffers和普通类属性不行。
```python
import torch 
import torch.nn as nn
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # （1）常见定义模型时的操作
        self.param_nn = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1, 1, 3, bias=False)),
            ('fc', nn.Linear(1, 2, bias=False))
        ]))

        # （2）使用register_buffer()定义一组参数
        self.register_buffer('param_buf', torch.randn(1, 2))

        # （3）使用形式类似的register_parameter()定义一组参数
        self.register_parameter('param_reg', nn.Parameter(torch.randn(1, 2)))

        # （4）按照类的属性形式定义一组变量
        self.param_attr = torch.randn(1, 2) 

        print('self._modules: ', self._modules)
        print('self._parameters: ', self._modules)
        print('self._buffers: ', self._modules)

    def forward(self, x):
        return x

net = Model()
```
只有(1)(3)会被`optimizer`更新.<br />模型实例化时，调用了 **init**() 方法，我们就可以看到调用输出结果：
```python
In [21]: net = Model()
self._modules:  OrderedDict([('param_nn', Sequential(   
    (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), bias=False)   
    (fc): Linear(in_features=1, out_features=2, bias=False) 
))]) 
self._parameters:  OrderedDict([('param_reg', Parameter containing: 
                                 tensor([[-0.5666, -0.2624]], requires_grad=True))]) 
self._buffers:  OrderedDict([('param_buf', tensor([[-0.4005, -0.8199]]))]) 
```
<br />`self._parameters` 和 `net.parameters()` 的返回值并不相同。<br />这里`self._parameters` 只记录了使用 `self.register_parameter()` 定义的参数，<br />而`net.parameters()` 返回所有可学习参数，<br />包括`self._modules` 中的参数和`self._parameters` 参数的并集。

# load_state_dict
`load_state_dict()` 是 PyTorch 中用于加载模型参数的函数，可以将保存的模型参数加载到模型中。通常在训练模型时，我们会将模型保存为一个文件，然后在需要使用模型时，使用 `load_state_dict()` 函数将模型参数加载到模型中。

`load_state_dict()` 函数的语法如下：

```python
model.load_state_dict(state_dict, strict=True)
```

其中，`model` 是需要加载模型参数的模型；`state_dict` 是一个字典，用于存储模型参数；`strict` 是一个布尔值，用于指定是否要求字典中的键与模型中的参数名称完全匹配。如果设置为 True，表示要求完全匹配；如果设置为 False，表示允许字典中的键缺失或多余。

以下是一个示例代码，展示了如何使用 `load_state_dict()` 函数加载模型参数：

```python
import torch.nn as nn

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

# 创建一个模型实例
model = MyModel()

# 保存模型参数到文件
torch.save(model.state_dict(), 'model.pth')

# 创建一个新的模型实例
new_model = MyModel()

# 加载保存的模型参数到新模型中
new_model.load_state_dict(torch.load('model.pth'))
```

在这个示例中，我们首先定义了一个简单的模型 `MyModel`，然后创建了一个模型实例 `model`。接着，我们使用 `torch.save()` 函数将模型参数保存到文件 `model.pth` 中。最后，我们创建了一个新的模型实例 `new_model`，并使用 `load_state_dict()` 函数将保存的模型参数加载到新模型中。

需要注意的是，当加载模型参数时，需要保证模型结构与保存时相同，否则加载可能会失败。另外，如果字典中的键与模型中的参数名称不完全匹配，且 `strict` 设置为 True，则会抛出异常。
# load
在 PyTorch 中，要保存和加载整个模型结构及参数，可以分别使用 `torch.save()` 和 `torch.load()` 函数。需要注意的是，在保存模型时，需要将模型的结构和参数一起保存，以便在加载模型时能够正确还原模型。

以下是一份示例代码，展示了如何保存和加载整个模型结构及参数：

```python
import torch
import torch.nn as nn

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# 创建一个模型实例
model = MyModel()

# 保存模型结构和参数到文件
torch.save(model.state_dict(), 'model.pth')
torch.save(model, 'model.pkl')

# 加载保存的模型结构和参数到新模型中
new_model1 = MyModel()
new_model1.load_state_dict(torch.load('model.pth'))

new_model2 = torch.load('model.pkl')
```

在这个示例中，我们首先定义了一个简单的模型 `MyModel`，然后创建了一个模型实例 `model`。接着，我们使用 `torch.save()` 函数将模型的参数保存到文件 `model.pth` 中，使用 `torch.save()` 函数将模型的结构和参数一起保存到文件 `model.pkl` 中。最后，我们创建了两个新的模型实例 `new_model1` 和 `new_model2`，并分别使用 `load_state_dict()` 函数和 `torch.load()` 函数将保存的模型结构和参数加载到新模型中。

需要注意的是，使用 `torch.save()` 函数保存模型结构和参数时，文件格式是二进制格式，不能直接读取和修改。如果需要修改模型参数，需要先将模型参数加载到内存中，然后进行修改。另外，在加载模型时，需要保证模型结构与保存时相同，否则加载可能会失败。
# zero_grad
`nn.Module.zero_grad()` 是 PyTorch 中的一个方法，用于将神经网络模块中所有参数的梯度设置为零。这通常在使用反向传播计算损失相对于网络参数的梯度之前完成。将梯度设置为零可以确保来自先前迭代的梯度不会累积并干扰当前迭代。

以下是 `zero_grad()` 的使用示例：

```python
import torch.nn as nn

model = nn.Linear(3, 2)  # 创建一个简单的线性模型
criterion = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 定义一个优化器

# 前向传递
inputs = torch.randn(2, 3)
outputs = model(inputs)
target = torch.randn(2, 2)

# 计算损失和梯度
loss = criterion(outputs, target)
loss.backward()

# 在下一次迭代之前将梯度清零
optimizer.zero_grad()

# 更新参数
optimizer.step()
```

在这个例子中，`zero_grad()` 在使用`loss.backward()`计算梯度之前被调用。这确保了梯度是新鲜的，没有被任何先前的迭代所影响。
# named_modules
`torch.Module.named_modules()` is a method in the PyTorch framework that returns an iterator over all modules in the current module hierarchy. It recursively traverses the module hierarchy and returns a tuple of the module's name and the module object.

Here is an example usage of `named_modules()`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = MyModel()

for name, module in model.named_modules():
    print(name, module)
```

In this example, `named_modules()` returns an iterator containing the tuples `( '', MyModel() )`, `( 'layer1', Linear(in_features=10, out_features=5, bias=True) )`, and `( 'layer2', Linear(in_features=5, out_features=1, bias=True) )`. The empty string `''` represents the root module.

