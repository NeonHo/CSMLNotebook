‘’’
import numpy as np

### 模拟参数 ###
boxes = np.array([[100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]],dtype = np.float)

### 定义函数 ###
def nms(dets, threshold):
    ### 用切片取出x1, y1, x2, y2及置信度 ###
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:,4]
    keep = [] ### 需要保留框的索引 ###
    index = scores.argsort()[::-1] ### 按置信度排序，index为一维列表，保留原始的索引 ###
    while len(index) > 0: ### 直到索引列表长度为0时停止 ###
        id = index[0] ### 置信度最大的总在indexp[0]处 ###
        keep.append(id) ### 把置信度最大的保留 ###
        index = np.delete(index,0,axis=0) ### 已经保留了 就从index列表中删除 ###
        delete = [] ### 保留要从index里删除的索引 ###
        ### 剩余的Bbox和置信度最大的逐个计算IoU ###
        for i in range(len(index)):
            j = index[i] 
            xx1=max(x1[id],x1[j])
            yy1=max(y1[id],y1[j])
            xx2=min(x2[id],x2[j])
            yy2=min(y2[id],y2[j])
            w=max(0,xx2-xx1+1)
            h=max(0,yy2-yy1+1)
            interface=w*h
            area =  (x2[id] - x1[id]) * (y2[id] - y1[id]) +  (x2[j] - x1[j]) * (y2[j] - y1[j]) - interface
            overlap=interface/area
            if overlap>=threshold: ### 记录IoU大于阈值的Bbox的位置 ###
                delete.append(i)
        ### 如果IoU大于阈值就从index列表里删除 ###
        index = np.delete(index,delete,axis=0)
    return keep
keep = nms(boxes, 0.7)
print(keep)
