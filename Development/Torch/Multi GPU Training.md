在 PyTorch 中，要使用多个 GPU 进行训练，你通常会使用 `torch.nn.DataParallel` 或 `torch.nn.parallel.DistributedDataParallel`。以下是两种方法的基本介绍和使用方式：

### 1. 使用 `DataParallel`

`DataParallel` 是一种相对简单的方式来实现模型在多个 GPU 上的并行训练。它会在每个 GPU 上复制模型，并将输入数据分割到各个 GPU 上，最后在进行反向传播时聚合梯度。

```python
import torch
import torch.nn as nn

# 定义你的模型
class MyModel(nn.Module):
    # ...

model = MyModel()
model = nn.DataParallel(model)

# 如果有多个 GPU，将模型转移到 GPU 上
if torch.cuda.is_available():
    model = model.cuda()

# 然后你可以像平时一样训练模型
```

使用 `DataParallel` 非常简单，但它可能不是最高效的方法，特别是在使用多个节点或大规模数据并行时。

### 2. 使用 `DistributedDataParallel`

`DistributedDataParallel`（DDP）提供了更复杂但更高效的并行处理方式。DDP 使用 PyTorch 的分布式通信包来在多个进程之间同步数据，适用于多 GPU 和多节点场景。

要使用 DDP，你需要稍微修改你的训练脚本来支持多进程运行，并确保每个进程只处理一部分数据。

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 初始化 PyTorch 分布式环境
    dist.init_process_group(backend='nccl')

    # 创建模型
    model = MyModel()
    model = model.cuda()
    model = DDP(model)

    # 然后像平时一样训练模型

if __name__ == "__main__":
    main()
```

使用 DDP 时，你需要使用 `torch.distributed.launch` 或者 `torch.distributed.run` 来启动多进程：

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE YOUR_SCRIPT.py
```

### 注意事项

- 使用多 GPU 训练时，确保你的数据集可以被均匀分割，以便每个 GPU 可以处理相等量的数据。
- 在多 GPU 训练中，模型的每个副本都会占用相同的 GPU 内存。因此，确保你的 GPU 内存足够大。
- `DataParallel` 通常在单个节点上工作得很好，但在跨节点场景中可能会遇到性能瓶颈。DDP 提供了更好的扩展性和效率。
- 当使用 DDP 时，需要注意数据的加载方式，确保每个进程加载的数据是不同的。

总之，选择哪种方法取决于你的具体需求，例如模型的大小、训练数据的大小、可用的 GPU 数量以及是否跨节点训练等。通常，对于大规模训练任务，DDP 是更优的选择。



# zhihu valid
pytorch多gpu并行训练 - link-web的文章 - 知乎
https://zhuanlan.zhihu.com/p/86441879
