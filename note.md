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
        momentum0.01: 10.9 15:20 (gpu1  tb 输入zscore了   3.76%  30/50epoch)
        momentum0.5: 10.9 17:50  (gpu1 init   输入进行了z-score normalize)
        momentum0.01: 10.9 18:12 (gpu1 train 输入进行了z-score normalize  )
                 0.5  10.9 20:00  gpu4 qua   3.88%  40epoch
                 
        momentum0.5 MCG03  2019-10-11-8:00 normalized  tanh (主要针对负的posmap的问题)
        
        
    (Attention)    
        momentum0.5 [attention] 2019-10-11-22:20  normalized tanh attention 修改了erase方式  attention的训练添加了crop  attentionlossrante=0.2  单卡
        
        momentum0.5 [train] 2019-10-12-9:30   normalized tanh attention  lossrate=0.03
        
        momentum0.5 [attention3] 2019-10-12-10:49 lossrate0.03 no clip  




