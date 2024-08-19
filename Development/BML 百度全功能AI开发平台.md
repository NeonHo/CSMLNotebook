# 数据标注
## 图像
### 图像分类的标注
#### 文件夹标注方式
一般在图像采集后，每个文件夹的名字为对应类别的名称；
文件夹下放置该类别的所有图片。
### 文件标注方式
`1.png` 对应的`1.json`:
`"labels": [("name": "aircraft")]`

### BML全功能AI开发平台

create database -> name: `{task_type}-{task_name}-{MMDD}`

- free memory -> platform memory
- Above the free memory -> BOS memory

import the configuration.
- local
- BOS directory
- share links
	- Baidu Pan
- Existing dataset from platform

Upload 22 images without labels, and then look up to the "质检报告".

EasyData工具：
Data cleaning -> backup the dataset V1
- remove the similar images (the similarity > 0.5). The similarity threshold needs to be greater.
- the blur threshold needs to be smaller.
- crop
- rotation
- mirror
- rotation correction
Add annotation
- If people can't recognize the image, delete it.
- 批量标注按钮一定要找到
![[Pasted image 20240819162744.png]]
![[Pasted image 20240819162838.png]]

Fine-tune pretrained model.

If we close the validation and testing set switch, the platform will automatically split your training set.

不平衡优化：imbalance adjustment.
增量训练：continue on the existing checkpoint.
智能归因：locate the reason for training effects.

