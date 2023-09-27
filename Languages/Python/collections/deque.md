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
