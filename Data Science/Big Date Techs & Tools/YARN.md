YARN（Yet Another Resource Negotiator）是一个**资源调度平台**，负责为运算程序提供服务器运算资源。
![[Pasted image 20250612192344.png]]
- ResourceManager（RM）：核心管理服务，负责资源的管理和分配。
- NodeManager（NM）：管理单个节点上的资源。
- [ApplicationMaster](https://zhida.zhihu.com/search?content_id=237287581&content_type=Article&match_order=1&q=ApplicationMaster&zhida_source=entity)（AM）：负责内部任务的资源申请和分配；任务的监控和容错。
- Container：容器，里面封装了任务运行所需要的资源。