# APRN
 
###现在的工作流程:
####Done:
    1.基本复现baseline: 同参数数量时68关键点2dNME=3.25% vs 3.36%  (论文报的3.269%)
        3d关键点的计算方式存疑,论文的原model只有6.20% 复现结果5.98% (论文抱的4.4%)
        今年4月arxiv上的一篇文章到了2.89%
    2.processor处理得到crop后的图片和posmap,同时做旋转增强保存下来,旋转与论文相同(+-45)
    3.训练时数据增强完全与论文相同,0.6~1.4的color scale,用random erase做遮挡
    4.将posmap分解为offset和mean,存储rotation translation scale参数,保存成4x4的矩阵
        从中解析出R(3) T(3) S(1)
       训练时回归这些参数和offset,然后转换合并成posmap
       分别计算param的loss和offset的loss,posmap加一个小权重的loss以防止陷入局部最优
       (理论上也可以不加其他loss,只算原来的loss,但没有参数监督可能不太能把param_regressor训练出来)
     R T S经过归一化,取值范围为[-1,1] [-1,1] [0,1] 
     因此分别用tanh tanh sigmoid作为激活函数
     
     若按照之前的设想,depth归一化到0,1带来了depth更难估计的问题,则用此方法只会带来T[2]
     难以收敛的结果,这很容易通过后处理解决
     (弱透视投影;depth减去了最小值导致网络其实很难确定何处是depth=0的点)
     现在的主要问题是能否训练好参数估计部分
     
     5.基于CBAM和distination的attention机制,都没有监督信息,前者略有提升,但当时的backbone并未达到
        baseline,后者没什么用
     
####Todo:
    0.metrics计算方式存疑,联系作者
    1.训练offset网络
    2.写一个pytorch版本,主要作用是通过require_grad控制param_regressor的梯度不影响encoder
      decoder
      另外,训练更快
    3.把random erase换成随机波动的噪声再erase
    4.加一个attention模块,用可见度作为监督信息训练
    5.训练时的可见度判定,给loss加attention
    (4,5都着力于attention)
    6.在实现可见度之后,引入inpaiting
    时间紧张啊......
    
    