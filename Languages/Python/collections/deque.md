```Python
import collections
q = collections.deque()
x = 0
q.append(x)  # deque/queue push
q.popleft()  # queue pop
# deque可以用索引访问，可以实现和peek相似的功能
q[-1]  # deque idx access.
len(q)  # 获取deque的长度, 可用于判断是否为空.
```
`pop()`，将队列尾部的数据弹出，并作为返回值。

`popleft()`，将队列头部的数据弹出，并作为返回值。

append(item)，添加一个数据到队列的尾部。与列表的append()方法功能相似。

appendleft(item)，添加一个数据到队列的头部。与append()的添加方向相反。