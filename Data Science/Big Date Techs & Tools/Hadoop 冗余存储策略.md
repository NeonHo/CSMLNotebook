
[[Hadoop]]采用冗余存储策略来确保数据可靠性和高可用性，主要通过以下机制实现：

### **1. 数据块复制（Block Replication）**
Hadoop分布式文件系统（[[HDFS]]）将文件分割成固定大小的块（默认128MB），每个块会复制多个副本（默认3个）。
- **策略特点**：
  - **跨机架存储**：副本会分布在不同机架的节点上，避免单个机架故障导致数据丢失。
  - **优先本地性**：第一个副本通常存储在客户端所在节点（若写入来自集群内），后续副本跨机架存储。
- **配置参数**：
  - `dfs.replication`：默认副本数（通常为3）。
  - `dfs.namenode.replication.min`：最小有效副本数。
  - `dfs.blocksize`：数据块大小。

### **2. 机架感知（Rack Awareness）**
Hadoop通过机架感知策略优化副本分布：
- **工作原理**：
  - 每个DataNode会被分配一个机架ID（如`/rack1`）。
  - NameNode根据机架ID决定副本位置，确保跨机架冗余。
- **典型分布**：
  - 副本1：客户端所在节点（或随机节点）。
  - 副本2：不同机架的随机节点。
  - 副本3：与副本2同机架的另一个节点（提高读取性能）。

### **3. 副本管理与修复**
Hadoop通过以下机制维护副本一致性：
- **心跳机制**：DataNode定期向NameNode发送心跳和块报告，NameNode监控副本状态。
- **副本不足处理**：
  - 当副本数低于`dfs.namenode.replication.min`时，NameNode启动复制任务。
  - 均衡器（Balancer）会定期优化集群内副本分布。
- **数据损坏修复**：
  - 客户端读取时通过CRC校验检测数据损坏。
  - 损坏块会被标记并从健康副本复制新副本。

### **4. 纠删码（Erasure Coding）**
Hadoop 3.0引入纠删码，相比传统复制更节省存储空间：
- **原理**：将数据块分成N个数据块和M个校验块，可容忍M个节点故障。
- **适用场景**：
  - 冷数据存储（如归档数据），存储成本降低50%以上。
  - 配置参数：`dfs.erasure.codec.name`（如RS-6-3）。

### **5. 联邦HDFS与高可用（HA）**
- **联邦HDFS**：通过多个NameNode管理不同命名空间，提高扩展性。
- **HA机制**：
  - 主备NameNode通过ZKFailoverController自动切换。
  - JournalNode集群同步编辑日志，确保元数据一致性。

### **优缺点**
- **优点**：
  - 高可靠性：容忍多节点故障。
  - 高可用性：读写操作不受单节点故障影响。
  - 负载均衡：副本分布优化读取性能。
- **缺点**：
  - 存储开销：3倍副本占用更多磁盘空间。
  - 写性能：多副本同步增加写入延迟。

### **配置示例**
在`hdfs-site.xml`中调整冗余参数：
```xml
<property>
  <name>dfs.replication</name>
  <value>3</value>
</property>
<property>
  <name>dfs.namenode.replication.min</name>
  <value>2</value>
</property>
<property>
  <name>dfs.blocksize</name>
  <value>134217728</value> <!-- 128MB -->
</property>
```

### **总结**
Hadoop通过多级冗余策略（块复制、机架感知、纠删码）实现数据可靠性，同时通过HA和联邦架构保证系统可用性，适用于大规模分布式环境下的海量数据存储。
