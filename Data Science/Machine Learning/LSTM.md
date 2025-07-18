
![[Pasted image 20250718103913.png]]

- 传送带 Conveyor Belt
	- 过去的信息传递到未来
	- Forget Gate
		- 通过与遗忘控制向量进行逐元素相乘，控制传递到过去信息以不同程度的保留向未来传递。
		- 遗忘控制向量是由当前时间步输入token与前一时间步输出state的concat，经过参数矩阵乘法，通过sigmoid激活得到。
	- Input Gate
		- 输入向量是由当前时间步输入token与前一时间步输出state的concat，经过参数矩阵乘法，通过tanh激活得到。
		- 输入控制向量控制输入向量以不同程度新增进入传送带，与经过遗忘门过滤后继续传送的过去信息进行相加。
			- 输入控制向量是由当前时间步输入token与前一时间步输出state的concat，经过参数矩阵乘法，通过sigmoid激活得到。
	- Output Gate
		- 经过Forget和Input后传送带上的信息将通过tanh激活输出。
		- 输出控制向量能够控制经过tanh激活后传送带上的信息以不同程度输出LSTM单元。
			- 输出控制向量是由当前时间步输入token与前一时间步输出state的concat，经过参数矩阵乘法，通过sigmoid激活得到。
