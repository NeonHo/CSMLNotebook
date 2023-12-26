`np.stack()` 是 NumPy 库中的一个函数，用于沿着新的轴堆叠数组序列（沿新的维度组合数组）。具体而言，`np.stack()` 接受一系列的数组，然后沿着指定的轴（新的维度）将它们堆叠在一起。

下面是 `np.stack()` 的简单示例：

```python
import numpy as np

# 定义两个数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 使用 stack 沿新的轴（维度）堆叠数组
stacked_array = np.stack([arr1, arr2])

# 输出堆叠后的数组
print(stacked_array)
```

输出：

```
[[1 2 3]
 [4 5 6]]
```

在这个例子中，`np.stack([arr1, arr2])` 将两个一维数组堆叠成一个二维数组，其中每个数组成为新数组的一行。

你还可以通过指定 `axis` 参数来沿着不同的轴进行堆叠。例如，`np.stack([arr1, arr2], axis=1)` 将两个一维数组沿着列的方向堆叠。

```python
stacked_array_axis1 = np.stack([arr1, arr2], axis=1)
print(stacked_array_axis1)
```

输出：

```
[[1 4]
 [2 5]
 [3 6]]
```

这里，两个一维数组沿着列的方向（axis=1）堆叠，形成一个新的二维数组。