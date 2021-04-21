基于深度学习根据叠前地震数据预测地层构造
======

1.环境
------
python3<br>
tensorflow2<br>
numpy<br>

2.目录
-----

experiments1：单分界面向斜背斜时域预测<br>
experiments2：4层地层，深度与速度预测结合还原地层<br>
experiments3：水流层预测<br>
experiments4：marmousi中高密度异常体预测<br>
<br>

###### 2.1下级目录：<br>
weights_em*:已训练好的权重<br>
model_em*:模型<br>
train_em*:训练<br>
predict_em*:预测<br>
<br>

3数据
--------

数据可以从链接：
https://pan.baidu.com/s/1vlnrjgjPd2j6jPBxoLslSg 
提取码：50wt 
网盘获得<br>
数据以numpy矩阵保存，已归一化，部分数据未切块，加载需要较大内存。
