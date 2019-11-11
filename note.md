 #PRN:
        Epoch30:    loss: 0.0111 - templateLoss: 0.0547 - mean_absolute_error: 0.0564 - val_loss: 0.0152 - val_templateLoss: 0.0748 - val_mean_absolute_error: 0.0646
        Epoch5:     loss: 0.0386 - templateLoss: 0.1895 - mean_absolute_error: 0.1131 - val_loss: 0.0412 - val_templateLoss: 0.2016 - val_mean_absolute_error: 0.1195


#APRN:
        l=1e-3(encode64)
        Epoch30:    loss: 0.3177 - templateLoss: 0.0444 - mean_absolute_error: 0.0569 - val_loss: 0.1090 - val_templateLoss: 0.0781 - val_mean_absolute_error: 0.0720
        Epoch5:     loss: 1.1835 - templateLoss: 0.1690 - mean_absolute_error: 0.1151 - val_loss: 0.9979 - val_templateLoss: 0.1608 - val_mean_absolute_error: 0.1098



#parameter:
        init:13353618
        initmy:13352633
        prnmy:32127209
###the number of BN parameters are not the same
        Total params: 13,372,445
        Trainable params: 13,360,555
        Non-trainable params: 11,890


### running log
    (with wightmask*16)
    (init)
        momentum0.01: 10.9 15:20 (gpu1  tb 输入zscore了   3.76%  30/50epoch)
        momentum0.5: 10.9 17:50  (gpu1 init   输入进行了z-score normalize)
        momentum0.01: 10.9 18:12 (gpu1 train 输入进行了z-score normalize  )
                 0.5  10.9 20:00  gpu4 qua   3.88%  40epoch
                 
        momentum0.5 MCG03 train  2019-10-11-8:00  normalized  tanh (主要针对负的posmap的问题)  [get:3.72]
        
                    MCG03  10-15-20:34  train    zeroz
        
                    [gpu07 train]2019-10-16-9-8-46+2019-10-17-10-19-52  initprn2 尝试复现结果 [get3.72]
        
     (qua)
        m0.5  MCG03 qua quaternion  lossrate 0 :1 :255:500 2019-10-13-15:00
            momentum0.5 [MG03 qua] 10-12-17:30 quaternion loss 0:1:500:500
         
    (Attention)    
        修改了erase方式  tanh 
        momentum0.5 [attention] 2019-10-13-03:00  normalized tanh attention   attention的训练  no clip  attentionlossrante=0.03  单卡 [get3.72  epoch32]
        
        momentum0.5 [train] 2019-10-13-15:30   normalized tanh attention  lossrate=1  [get3.75]
        
        momentum0.5 [attention3] 2019-10-13-15:49 lossrate1 no clip   [get bad]

        m0.5 [attention2]  10-15-13:28+2019-10-17-10-29-16  lossrate0.1  noclip  l2rate=0.0001 [get3.68]
        


    10.28 lr5e-4 batchsize 32比16略好   10轮下降0.1不可取
    
    600blocks:
    10.31 lr1e-5warmup siam
    11.1 9:55 1r1e-5warmup init
    
    630blocks
    11.1 14:29 lr1e-5 init 
    11.3 23:00 attentionbatch16/48   10:35 siam
    
 
    11.5
    晚上三组 attention batch48 l2=0.0001 lossrate=0.1
             完全体  visible
             attention batch48 l2=0.0001 lossrate=0.5
             
             
     visible1 batchsize48 比visible稍微下降0.01基本没问题
     visible2 finalposerate 0.01 性能下降
     
     
    11.11上午两组SDN
    (visible2)
        self.criterion0 = getLossFunction('fwrse')(0.1)  # final pos
        self.criterion1 = getLossFunction('fwrse')(0.5)  # offset
        self.criterion2 = getLossFunction('fwrse')(1)  # kpt
        self.criterion3 = getLossFunction('bce')(0.1)  # attention
        self.criterion4 = getLossFunction('smooth')(0.)
        self.metrics0 = getLossFunction('nme')(1.)
        self.metrics1 = getLossFunction('frse')(1.)
        self.metrics2 = getLossFunction('kptc')(1.)
        self.metrics3 = getLossFunction('mae')(1.)
     
    (visible1)
        decay 0.0002