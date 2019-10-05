import time
import os

start_time = time.time()
cmd = "python torchrun.py --batchSize=16 --isTrain=True --isTest=False --gpu=4 --visibleDevice=4,5,6,7 --epoch=30 -struct=QuaternionOffsetPRN -td data/images/300W_LP-crop " \
      "data/images/300W-3D-crop -vd data/images/AFLW2000-crop"
# cmd = "python testcmd.py"
while True:
    time.sleep(10)
    now_time = time.time()
    if now_time - start_time > 9000:
        break
    else:
        print(now_time-start_time)
os.system(cmd)
