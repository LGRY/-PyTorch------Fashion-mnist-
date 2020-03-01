# 动手学深度学习PyTorch版--大作业--Fashion-mnist分类任务
目前最高得分为 **0.9547**

## 背景介绍
参考【我的博客地址】https://blog.csdn.net/sinat_29950703/article/details/104589911
## 目录
准备4个文件夹  
- raw_data: 用于存放ubyte.gz原文件，解压完也在此文件夹下  
- mnist_train: 存放训练集的0-9共10个文件夹  
- mnist_test: 存放测试集的0-9共10个文件  
- model: 存放训练好的h5模型  
## 1.文件预处理
### 1.1 解压ubyte文件
- window上:  直接解压
- linux上:  gunzip xx.gz  
    ```python
    import os  
    tar_str1 = 'gunzip ./train-images-idx3-ubyte.gz'    
    tar_str2 = 'gunzip ./train-labels-idx1-ubyte.gz'   
    tar_str3 = 'gunzip ./t10k-images-idx3-ubyte.gz'  
    tar_str4 = 'gunzip ./t10k-labels-idx1-ubyte.gz'  
    os.system(tar_str1)  
    os.system(tar_str2)  
    os.system(tar_str3)   
    os.system(tar_str4)  
    ```
### 1.2 将训练集的ubyte文件转化为图片
将Fashion-mnist训练集的10个种类分在0-9共10个文件夹下  
详见ubyte_to_img_train.py
### 1.3 将测试集的ubyte文件转化为图片
将Fashion-mnist测试集的10个种类分在0-9共10个文件夹下    
详见ubyte_to_img_test.py
## 2.数据准备
### 2.1调整输入的归一化系数
计算整个数据集的归一化系数
```python
# 求整个数据集的均值
temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    channel_mean = torch.mean(X, dim=(0,2,3))  # 按channel求均值(不过这里只有1个channel)
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt
print('整个数据集的像素均值:{}'.format(dataset_global_mean))
# 求整个数据集的标准差
cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))
'''
运行结果：

计算数据集均值标准差
整个数据集的像素均值:0.2860483762389695
整个数据集的像素标准差:0.3529184201347597
'''
```  
### 2.2进行数据增强
【图像尺寸】Resize到224x224  
【数据增强】RandomHorizontalFlip

### 2.3 尝试三种优化方法
```python
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
optimizer = optim.RMSprop(model.parameters(), lr = lr,alpha=0.9)
```
### 2.4 添加学习率阶段性下降的策略
【哦豁】这种方法忘记尝试了 3月1日24时到截止时间 来不及咯 以后再试吧
```python
def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, lr, lr_period, lr_decay):
    ...
    ... 
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:  # 每lr_period个epoch，学习率衰减一次
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            ...
            ...
```

## 3.训练和测试同时进行
【网络模型】ResNet18的迁移学习  
打印结果预览
```python
training on  cuda:0
epoch 1, loss 0.3812, train acc 0.8726, test acc 0.9125, time 194.0 sec
find best! save at model/best.pth
epoch 2, loss 0.2112, train acc 0.9255, test acc 0.9222, time 192.7 sec
find best! save at model/best.pth
epoch 3, loss 0.1737, train acc 0.9389, test acc 0.9288, time 187.0 sec
find best! save at model/best.pth
epoch 4, loss 0.1478, train acc 0.9481, test acc 0.9346, time 190.8 sec
find best! save at model/best.pth
...
...
epoch 113, loss 0.0006, train acc 1.0000, test acc 0.9512, time 189.4 sec
epoch 114, loss 0.0006, train acc 1.0000, test acc 0.9528, time 186.4 sec
find best! save at model/best.pth
...
...
epoch 180, loss 0.0011, train acc 1.0000, test acc 0.9547, time 173.8 sec
find best! save at model/best.pth
```  
【注】待3月1日24时后，再上传完整的训练代码和scv文件
