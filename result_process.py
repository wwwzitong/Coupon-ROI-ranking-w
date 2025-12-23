import re
import pandas as pd

# 将你的日志内容粘贴到这里
log_data = """
--- 运行配置 ---
Model Class: res_base_DFCL
Model Path: ./model/res_base_DFCL_4pll_2pos_gradient_alpha=2.0
Alpha: 2.0
--------------------
策略中的副本数: 1
Epoch 1/50
  6/500 [..............................] - ETA: 6s - total_loss: 5484.0036 - weighted_task_loss: 2737.6142 - decision_loss: -8.7752 - paid_loss: 1622.3772 - cost_loss: 1115.2370     WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0114s vs `on_train_batch_end` time: 0.0534s). Check your callbacks.
498/500 [============================>.] - ETA: 0s - total_loss: 2248.3921 - weighted_task_loss: 1121.5697 - decision_loss: -5.2527 - paid_loss: 672.9491 - cost_loss: 448.6206  
Epoch 1 metrics: {'total_loss': 2915.48046875, 'weighted_task_loss': 1455.574951171875, 'decision_loss': -4.330565929412842, 'paid_loss': 1080.5228271484375, 'cost_loss': 375.05206298828125, 'val_total_loss': 1892.892822265625, 'val_weighted_task_loss': 943.0914306640625, 'val_paid_loss': 511.08160400390625, 'val_cost_loss': 432.00982666015625, 'val_decision_loss': -6.709957122802734}       
500/500 [==============================] - 37s 48ms/step - total_loss: 2248.7965 - weighted_task_loss: 1121.7727 - decision_loss: -5.2511 - paid_loss: 673.3159 - cost_loss: 448.4568 - val_total_loss: 1892.8928 - val_weighted_task_loss: 943.0914 - val_paid_loss: 511.0816 - val_cost_loss: 432.0098 - val_decision_loss: -6.7100
Epoch 2/50
498/500 [============================>.] - ETA: 0s - total_loss: 1711.3609 - weighted_task_loss: 853.0416 - decision_loss: -5.2777 - paid_loss: 447.4496 - cost_loss: 405.5920 
Epoch 2 metrics: {'total_loss': 911.6315307617188, 'weighted_task_loss': 451.619384765625, 'decision_loss': -8.39277172088623, 'paid_loss': 32.41022872924805, 'cost_loss': 419.20916748046875, 'val_total_loss': 1877.39013671875, 'val_weighted_task_loss': 935.3991088867188, 'val_paid_loss': 499.453369140625, 'val_cost_loss': 435.94573974609375, 'val_decision_loss': -6.591961860656738}
500/500 [==============================] - 28s 56ms/step - total_loss: 1710.1173 - weighted_task_loss: 852.4143 - decision_loss: -5.2888 - paid_loss: 446.8833 - cost_loss: 405.5310 - val_total_loss: 1877.3901 - val_weighted_task_loss: 935.3991 - val_paid_loss: 499.4534 - val_cost_loss: 435.9457 - val_decision_loss: -6.5920
Epoch 3/50
500/500 [==============================] - ETA: 0s - total_loss: 1772.7846 - weighted_task_loss: 883.8983 - decision_loss: -4.9880 - paid_loss: 484.1454 - cost_loss: 399.7529
Epoch 3 metrics: {'total_loss': 775.2405395507812, 'weighted_task_loss': 385.2022705078125, 'decision_loss': -4.835975646972656, 'paid_loss': 37.731658935546875, 'cost_loss': 347.4706115722656, 'val_total_loss': 1883.3690185546875, 'val_weighted_task_loss': 938.3763427734375, 'val_paid_loss': 503.3053283691406, 'val_cost_loss': 435.071044921875, 'val_decision_loss': -6.616314888000488}      
500/500 [==============================] - 30s 59ms/step - total_loss: 1770.7935 - weighted_task_loss: 882.9029 - decision_loss: -4.9877 - paid_loss: 483.2544 - cost_loss: 399.6485 - val_total_loss: 1883.3690 - val_weighted_task_loss: 938.3763 - val_paid_loss: 503.3053 - val_cost_loss: 435.0710 - val_decision_loss: -6.6163
Epoch 4/50
500/500 [==============================] - ETA: 0s - total_loss: 1664.2936 - weighted_task_loss: 829.5214 - decision_loss: -5.2508 - paid_loss: 428.3635 - cost_loss: 401.1579
Epoch 4 metrics: {'total_loss': 1722.4993896484375, 'weighted_task_loss': 858.1505126953125, 'decision_loss': -6.198314189910889, 'paid_loss': 510.4731750488281, 'cost_loss': 347.6773681640625, 'val_total_loss': 1869.966796875, 'val_weighted_task_loss': 931.682373046875, 'val_paid_loss': 502.3203430175781, 'val_cost_loss': 429.36199951171875, 'val_decision_loss': -6.602056980133057}
500/500 [==============================] - 30s 60ms/step - total_loss: 1664.4098 - weighted_task_loss: 829.5785 - decision_loss: -5.2527 - paid_loss: 428.5274 - cost_loss: 401.0511 - val_total_loss: 1869.9668 - val_weighted_task_loss: 931.6824 - val_paid_loss: 502.3203 - val_cost_loss: 429.3620 - val_decision_loss: -6.6021
Epoch 5/50
499/500 [============================>.] - ETA: 0s - total_loss: 1647.9134 - weighted_task_loss: 821.4505 - decision_loss: -5.0125 - paid_loss: 425.7228 - cost_loss: 395.7277 
Epoch 5 metrics: {'total_loss': 1697.2586669921875, 'weighted_task_loss': 847.4228515625, 'decision_loss': -2.4129369258880615, 'paid_loss': 521.8397827148438, 'cost_loss': 325.5830993652344, 'val_total_loss': 1879.3294677734375, 'val_weighted_task_loss': 936.3491821289062, 'val_paid_loss': 508.3840637207031, 'val_cost_loss': 427.9651184082031, 'val_decision_loss': -6.631086826324463}       
500/500 [==============================] - 30s 60ms/step - total_loss: 1648.1104 - weighted_task_loss: 821.5541 - decision_loss: -5.0021 - paid_loss: 426.1065 - cost_loss: 395.4477 - val_total_loss: 1879.3295 - val_weighted_task_loss: 936.3492 - val_paid_loss: 508.3841 - val_cost_loss: 427.9651 - val_decision_loss: -6.6311
Epoch 6/50
500/500 [==============================] - ETA: 0s - total_loss: 1602.4685 - weighted_task_loss: 798.6641 - decision_loss: -5.1403 - paid_loss: 406.6005 - cost_loss: 392.0636 
Epoch 6 metrics: {'total_loss': 2652.526123046875, 'weighted_task_loss': 1324.913818359375, 'decision_loss': -2.6985416412353516, 'paid_loss': 1016.748046875, 'cost_loss': 308.1657409667969, 'val_total_loss': 1894.0421142578125, 'val_weighted_task_loss': 943.717041015625, 'val_paid_loss': 508.7169189453125, 'val_cost_loss': 435.0001525878906, 'val_decision_loss': -6.607974529266357}
500/500 [==============================] - 30s 60ms/step - total_loss: 1604.5644 - weighted_task_loss: 799.7145 - decision_loss: -5.1354 - paid_loss: 407.8183 - cost_loss: 391.8962 - val_total_loss: 1894.0421 - val_weighted_task_loss: 943.7170 - val_paid_loss: 508.7169 - val_cost_loss: 435.0002 - val_decision_loss: -6.6080
Epoch 7/50
499/500 [============================>.] - ETA: 0s - total_loss: 1697.1304 - weighted_task_loss: 845.9705 - decision_loss: -5.1894 - paid_loss: 443.7433 - cost_loss: 402.2272
Epoch 7 metrics: {'total_loss': 2019.5179443359375, 'weighted_task_loss': 1007.8900146484375, 'decision_loss': -3.737865924835205, 'paid_loss': 534.91943359375, 'cost_loss': 472.9705810546875, 'val_total_loss': 1907.5030517578125, 'val_weighted_task_loss': 950.44189453125, 'val_paid_loss': 514.7333374023438, 'val_cost_loss': 435.7085876464844, 'val_decision_loss': -6.619222164154053}        
500/500 [==============================] - 30s 60ms/step - total_loss: 1698.4174 - weighted_task_loss: 846.6169 - decision_loss: -5.1836 - paid_loss: 444.1073 - cost_loss: 402.5096 - val_total_loss: 1907.5031 - val_weighted_task_loss: 950.4419 - val_paid_loss: 514.7333 - val_cost_loss: 435.7086 - val_decision_loss: -6.6192
Epoch 8/50
499/500 [============================>.] - ETA: 0s - total_loss: 1644.1445 - weighted_task_loss: 819.5268 - decision_loss: -5.0909 - paid_loss: 428.9688 - cost_loss: 390.5580  
Epoch 8 metrics: {'total_loss': 2279.93603515625, 'weighted_task_loss': 1133.922119140625, 'decision_loss': -12.091791152954102, 'paid_loss': 569.1366577148438, 'cost_loss': 564.7855224609375, 'val_total_loss': 1875.7659912109375, 'val_weighted_task_loss': 934.569091796875, 'val_paid_loss': 502.29400634765625, 'val_cost_loss': 432.2751159667969, 'val_decision_loss': -6.627835750579834}      
500/500 [==============================] - 29s 59ms/step - total_loss: 1646.6826 - weighted_task_loss: 820.7819 - decision_loss: -5.1189 - paid_loss: 429.5284 - cost_loss: 391.2535 - val_total_loss: 1875.7660 - val_weighted_task_loss: 934.5691 - val_paid_loss: 502.2940 - val_cost_loss: 432.2751 - val_decision_loss: -6.6278
Epoch 9/50
498/500 [============================>.] - ETA: 0s - total_loss: 1693.8357 - weighted_task_loss: 844.2800 - decision_loss: -5.2758 - paid_loss: 444.7658 - cost_loss: 399.5142 
Epoch 9 metrics: {'total_loss': 2250.685791015625, 'weighted_task_loss': 1123.2880859375, 'decision_loss': -4.109542369842529, 'paid_loss': 638.831298828125, 'cost_loss': 484.456787109375, 'val_total_loss': 1885.7310791015625, 'val_weighted_task_loss': 939.55029296875, 'val_paid_loss': 509.02215576171875, 'val_cost_loss': 430.52813720703125, 'val_decision_loss': -6.6305131912231445}
500/500 [==============================] - 29s 59ms/step - total_loss: 1694.2802 - weighted_task_loss: 844.5032 - decision_loss: -5.2739 - paid_loss: 444.7375 - cost_loss: 399.7657 - val_total_loss: 1885.7311 - val_weighted_task_loss: 939.5503 - val_paid_loss: 509.0222 - val_cost_loss: 430.5281 - val_decision_loss: -6.6305
Epoch 10/50
499/500 [============================>.] - ETA: 0s - total_loss: 1592.8081 - weighted_task_loss: 793.7363 - decision_loss: -5.3355 - paid_loss: 393.1654 - cost_loss: 400.5710  
Epoch 10 metrics: {'total_loss': 888.7261962890625, 'weighted_task_loss': 442.5556640625, 'decision_loss': -3.614851474761963, 'paid_loss': 31.566129684448242, 'cost_loss': 410.9895324707031, 'val_total_loss': 1876.13427734375, 'val_weighted_task_loss': 934.7491455078125, 'val_paid_loss': 503.3828125, 'val_cost_loss': 431.3663635253906, 'val_decision_loss': -6.635948181152344}
500/500 [==============================] - 29s 59ms/step - total_loss: 1589.9974 - weighted_task_loss: 792.3344 - decision_loss: -5.3286 - paid_loss: 391.7218 - cost_loss: 400.6126 - val_total_loss: 1876.1343 - val_weighted_task_loss: 934.7491 - val_paid_loss: 503.3828 - val_cost_loss: 431.3664 - val_decision_loss: -6.6359
Epoch 11/50
500/500 [==============================] - ETA: 0s - total_loss: 1716.6479 - weighted_task_loss: 855.8152 - decision_loss: -5.0175 - paid_loss: 460.3505 - cost_loss: 395.4647   
Epoch 11 metrics: {'total_loss': 1544.5174560546875, 'weighted_task_loss': 771.634033203125, 'decision_loss': -1.2494488954544067, 'paid_loss': 518.3361206054688, 'cost_loss': 253.29788208007812, 'val_total_loss': 1890.3614501953125, 'val_weighted_task_loss': 941.8650512695312, 'val_paid_loss': 511.544189453125, 'val_cost_loss': 430.32086181640625, 'val_decision_loss': -6.631314754486084}   
500/500 [==============================] - 29s 58ms/step - total_loss: 1716.3044 - weighted_task_loss: 855.6472 - decision_loss: -5.0100 - paid_loss: 460.4662 - cost_loss: 395.1810 - val_total_loss: 1890.3615 - val_weighted_task_loss: 941.8651 - val_paid_loss: 511.5442 - val_cost_loss: 430.3209 - val_decision_loss: -6.6313
Epoch 12/50
500/500 [==============================] - ETA: 0s - total_loss: 1587.9918 - weighted_task_loss: 791.3206 - decision_loss: -5.3507 - paid_loss: 393.8113 - cost_loss: 397.5093
Epoch 12 metrics: {'total_loss': 1839.0648193359375, 'weighted_task_loss': 917.3209228515625, 'decision_loss': -4.422999858856201, 'paid_loss': 521.5853271484375, 'cost_loss': 395.735595703125, 'val_total_loss': 1892.5679931640625, 'val_weighted_task_loss': 942.9752197265625, 'val_paid_loss': 512.4182739257812, 'val_cost_loss': 430.5569763183594, 'val_decision_loss': -6.6175103187561035}    
500/500 [==============================] - 29s 59ms/step - total_loss: 1588.4930 - weighted_task_loss: 791.5721 - decision_loss: -5.3489 - paid_loss: 394.0663 - cost_loss: 397.5058 - val_total_loss: 1892.5680 - val_weighted_task_loss: 942.9752 - val_paid_loss: 512.4183 - val_cost_loss: 430.5570 - val_decision_loss: -6.6175
Epoch 13/50
498/500 [============================>.] - ETA: 0s - total_loss: 1717.7095 - weighted_task_loss: 856.2192 - decision_loss: -5.2712 - paid_loss: 451.2534 - cost_loss: 404.9657  
Epoch 13 metrics: {'total_loss': 2041.8785400390625, 'weighted_task_loss': 1015.71630859375, 'decision_loss': -10.44589614868164, 'paid_loss': 515.3511352539062, 'cost_loss': 500.36517333984375, 'val_total_loss': 1889.4500732421875, 'val_weighted_task_loss': 941.41357421875, 'val_paid_loss': 508.5716857910156, 'val_cost_loss': 432.8419189453125, 'val_decision_loss': -6.62289571762085}       
500/500 [==============================] - 29s 58ms/step - total_loss: 1716.5829 - weighted_task_loss: 855.6490 - decision_loss: -5.2850 - paid_loss: 450.6810 - cost_loss: 404.9680 - val_total_loss: 1889.4501 - val_weighted_task_loss: 941.4136 - val_paid_loss: 508.5717 - val_cost_loss: 432.8419 - val_decision_loss: -6.6229
Epoch 14/50
500/500 [==============================] - ETA: 0s - total_loss: 1608.7322 - weighted_task_loss: 801.7706 - decision_loss: -5.1910 - paid_loss: 409.1277 - cost_loss: 392.6429  
Epoch 14 metrics: {'total_loss': 1103.9541015625, 'weighted_task_loss': 548.6005249023438, 'decision_loss': -6.753021240234375, 'paid_loss': 32.404396057128906, 'cost_loss': 516.1961059570312, 'val_total_loss': 1865.380126953125, 'val_weighted_task_loss': 929.378173828125, 'val_paid_loss': 502.4469909667969, 'val_cost_loss': 426.93115234375, 'val_decision_loss': -6.623786449432373}
500/500 [==============================] - 29s 58ms/step - total_loss: 1607.7246 - weighted_task_loss: 801.2653 - decision_loss: -5.1941 - paid_loss: 408.3758 - cost_loss: 392.8895 - val_total_loss: 1865.3801 - val_weighted_task_loss: 929.3782 - val_paid_loss: 502.4470 - val_cost_loss: 426.9312 - val_decision_loss: -6.6238
Epoch 15/50
499/500 [============================>.] - ETA: 0s - total_loss: 1617.1514 - weighted_task_loss: 806.0939 - decision_loss: -4.9636 - paid_loss: 424.4090 - cost_loss: 381.6849
Epoch 15 metrics: {'total_loss': 1224.453125, 'weighted_task_loss': 608.1854858398438, 'decision_loss': -8.082127571105957, 'paid_loss': 55.89497375488281, 'cost_loss': 552.29052734375, 'val_total_loss': 1885.5572509765625, 'val_weighted_task_loss': 939.4697265625, 'val_paid_loss': 510.3498840332031, 'val_cost_loss': 429.1198425292969, 'val_decision_loss': -6.617820739746094}
500/500 [==============================] - 29s 58ms/step - total_loss: 1615.5837 - weighted_task_loss: 805.3038 - decision_loss: -4.9761 - paid_loss: 422.9379 - cost_loss: 382.3660 - val_total_loss: 1885.5573 - val_weighted_task_loss: 939.4697 - val_paid_loss: 510.3499 - val_cost_loss: 429.1198 - val_decision_loss: -6.6178
Epoch 16/50
500/500 [==============================] - ETA: 0s - total_loss: 1772.5332 - weighted_task_loss: 883.7621 - decision_loss: -5.0089 - paid_loss: 486.1306 - cost_loss: 397.6315
Epoch 16 metrics: {'total_loss': 1017.5034790039062, 'weighted_task_loss': 506.4218444824219, 'decision_loss': -4.659811019897461, 'paid_loss': 44.003414154052734, 'cost_loss': 462.4184265136719, 'val_total_loss': 1882.2481689453125, 'val_weighted_task_loss': 937.8065185546875, 'val_paid_loss': 504.72900390625, 'val_cost_loss': 433.0775451660156, 'val_decision_loss': -6.635167121887207}     
500/500 [==============================] - 29s 59ms/step - total_loss: 1771.0261 - weighted_task_loss: 883.0090 - decision_loss: -5.0082 - paid_loss: 485.2481 - cost_loss: 397.7609 - val_total_loss: 1882.2482 - val_weighted_task_loss: 937.8065 - val_paid_loss: 504.7290 - val_cost_loss: 433.0775 - val_decision_loss: -6.6352
Epoch 17/50
500/500 [==============================] - ETA: 0s - total_loss: 1626.1359 - weighted_task_loss: 810.4966 - decision_loss: -5.1428 - paid_loss: 418.4110 - cost_loss: 392.0855   
Epoch 17 metrics: {'total_loss': 1950.173095703125, 'weighted_task_loss': 971.7332763671875, 'decision_loss': -6.706544399261475, 'paid_loss': 516.1827392578125, 'cost_loss': 455.550537109375, 'val_total_loss': 1873.0162353515625, 'val_weighted_task_loss': 933.19873046875, 'val_paid_loss': 503.3349609375, 'val_cost_loss': 429.8638000488281, 'val_decision_loss': -6.6187615394592285}
500/500 [==============================] - 29s 58ms/step - total_loss: 1626.7827 - weighted_task_loss: 810.8184 - decision_loss: -5.1459 - paid_loss: 418.6062 - cost_loss: 392.2122 - val_total_loss: 1873.0162 - val_weighted_task_loss: 933.1987 - val_paid_loss: 503.3350 - val_cost_loss: 429.8638 - val_decision_loss: -6.6188
Epoch 18/50
499/500 [============================>.] - ETA: 0s - total_loss: 1689.4600 - weighted_task_loss: 842.1695 - decision_loss: -5.1210 - paid_loss: 448.6347 - cost_loss: 393.5347
Epoch 18 metrics: {'total_loss': 885.6535034179688, 'weighted_task_loss': 439.52239990234375, 'decision_loss': -6.608697414398193, 'paid_loss': 37.72228240966797, 'cost_loss': 401.80010986328125, 'val_total_loss': 1872.42724609375, 'val_weighted_task_loss': 932.9085693359375, 'val_paid_loss': 507.75433349609375, 'val_cost_loss': 425.1542053222656, 'val_decision_loss': -6.610062599182129}    
500/500 [==============================] - 29s 58ms/step - total_loss: 1686.2512 - weighted_task_loss: 840.5621 - decision_loss: -5.1270 - paid_loss: 446.9944 - cost_loss: 393.5677 - val_total_loss: 1872.4272 - val_weighted_task_loss: 932.9086 - val_paid_loss: 507.7543 - val_cost_loss: 425.1542 - val_decision_loss: -6.6101
Epoch 19/50
498/500 [============================>.] - ETA: 0s - total_loss: 1615.5850 - weighted_task_loss: 805.1903 - decision_loss: -5.2044 - paid_loss: 414.4424 - cost_loss: 390.7478 
Epoch 19 metrics: {'total_loss': 1125.9326171875, 'weighted_task_loss': 560.3482055664062, 'decision_loss': -5.2362470626831055, 'paid_loss': 40.99152374267578, 'cost_loss': 519.356689453125, 'val_total_loss': 1885.1639404296875, 'val_weighted_task_loss': 939.26806640625, 'val_paid_loss': 505.9100341796875, 'val_cost_loss': 433.3580322265625, 'val_decision_loss': -6.627813816070557}
500/500 [==============================] - 30s 60ms/step - total_loss: 1612.6299 - weighted_task_loss: 803.7113 - decision_loss: -5.2073 - paid_loss: 412.2066 - cost_loss: 391.5047 - val_total_loss: 1885.1639 - val_weighted_task_loss: 939.2681 - val_paid_loss: 505.9100 - val_cost_loss: 433.3580 - val_decision_loss: -6.6278
Epoch 20/50
500/500 [==============================] - ETA: 0s - total_loss: 1663.8038 - weighted_task_loss: 829.3022 - decision_loss: -5.1995 - paid_loss: 427.9348 - cost_loss: 401.3674 
Epoch 20 metrics: {'total_loss': 1128.597900390625, 'weighted_task_loss': 560.4307250976562, 'decision_loss': -7.73647403717041, 'paid_loss': 40.919429779052734, 'cost_loss': 519.5112915039062, 'val_total_loss': 1886.75048828125, 'val_weighted_task_loss': 940.0616455078125, 'val_paid_loss': 509.1185607910156, 'val_cost_loss': 430.943115234375, 'val_decision_loss': -6.627180576324463}        
500/500 [==============================] - 29s 59ms/step - total_loss: 1662.7355 - weighted_task_loss: 828.7655 - decision_loss: -5.2046 - paid_loss: 427.1623 - cost_loss: 401.6032 - val_total_loss: 1886.7505 - val_weighted_task_loss: 940.0616 - val_paid_loss: 509.1186 - val_cost_loss: 430.9431 - val_decision_loss: -6.6272
Epoch 21/50
498/500 [============================>.] - ETA: 0s - total_loss: 1705.2534 - weighted_task_loss: 849.9420 - decision_loss: -5.3693 - paid_loss: 437.6974 - cost_loss: 412.2446 
Epoch 21 metrics: {'total_loss': 571.371826171875, 'weighted_task_loss': 283.1098327636719, 'decision_loss': -5.152188777923584, 'paid_loss': 36.6419677734375, 'cost_loss': 246.46786499023438, 'val_total_loss': 1872.74560546875, 'val_weighted_task_loss': 933.0586547851562, 'val_paid_loss': 504.5802917480469, 'val_cost_loss': 428.4783630371094, 'val_decision_loss': -6.628276348114014}        
500/500 [==============================] - 29s 59ms/step - total_loss: 1699.4453 - weighted_task_loss: 847.0372 - decision_loss: -5.3708 - paid_loss: 435.3027 - cost_loss: 411.7345 - val_total_loss: 1872.7456 - val_weighted_task_loss: 933.0587 - val_paid_loss: 504.5803 - val_cost_loss: 428.4784 - val_decision_loss: -6.6283
Epoch 22/50
498/500 [============================>.] - ETA: 0s - total_loss: 1678.3206 - weighted_task_loss: 836.5654 - decision_loss: -5.1897 - paid_loss: 435.2449 - cost_loss: 401.3205
Epoch 22 metrics: {'total_loss': 3891.28173828125, 'weighted_task_loss': 1944.222900390625, 'decision_loss': -2.835956573486328, 'paid_loss': 1481.360107421875, 'cost_loss': 462.86273193359375, 'val_total_loss': 1893.4312744140625, 'val_weighted_task_loss': 943.4015502929688, 'val_paid_loss': 511.0575256347656, 'val_cost_loss': 432.3440246582031, 'val_decision_loss': -6.62814474105835}      
500/500 [==============================] - 26s 53ms/step - total_loss: 1685.0969 - weighted_task_loss: 839.9609 - decision_loss: -5.1751 - paid_loss: 438.6624 - cost_loss: 401.2985 - val_total_loss: 1893.4313 - val_weighted_task_loss: 943.4016 - val_paid_loss: 511.0575 - val_cost_loss: 432.3440 - val_decision_loss: -6.6281
Epoch 23/50
499/500 [============================>.] - ETA: 0s - total_loss: 1609.6326 - weighted_task_loss: 802.3958 - decision_loss: -4.8410 - paid_loss: 415.9966 - cost_loss: 386.3992 
Epoch 23 metrics: {'total_loss': 1859.2691650390625, 'weighted_task_loss': 925.7503662109375, 'decision_loss': -7.768399715423584, 'paid_loss': 520.8544311523438, 'cost_loss': 404.89593505859375, 'val_total_loss': 1877.781005859375, 'val_weighted_task_loss': 935.5789794921875, 'val_paid_loss': 507.7850036621094, 'val_cost_loss': 427.7939758300781, 'val_decision_loss': -6.623055934906006}    
500/500 [==============================] - 26s 52ms/step - total_loss: 1610.6292 - weighted_task_loss: 802.8882 - decision_loss: -4.8527 - paid_loss: 416.4152 - cost_loss: 386.4730 - val_total_loss: 1877.7810 - val_weighted_task_loss: 935.5790 - val_paid_loss: 507.7850 - val_cost_loss: 427.7940 - val_decision_loss: -6.6231
Epoch 24/50
499/500 [============================>.] - ETA: 0s - total_loss: 1582.2479 - weighted_task_loss: 788.5529 - decision_loss: -5.1422 - paid_loss: 390.0792 - cost_loss: 398.4737
Epoch 24 metrics: {'total_loss': 1721.9635009765625, 'weighted_task_loss': 858.8400268554688, 'decision_loss': -4.283426761627197, 'paid_loss': 504.28436279296875, 'cost_loss': 354.5556640625, 'val_total_loss': 1869.0404052734375, 'val_weighted_task_loss': 931.2081298828125, 'val_paid_loss': 501.7230529785156, 'val_cost_loss': 429.4850769042969, 'val_decision_loss': -6.624090671539307}      
500/500 [==============================] - 27s 53ms/step - total_loss: 1582.8056 - weighted_task_loss: 788.8334 - decision_loss: -5.1387 - paid_loss: 390.5351 - cost_loss: 398.2984 - val_total_loss: 1869.0404 - val_weighted_task_loss: 931.2081 - val_paid_loss: 501.7231 - val_cost_loss: 429.4851 - val_decision_loss: -6.6241
训练完成，正在将模型保存到: ./model/res_base_DFCL_4pll_2pos_gradient_alpha=2.0
"""

# 1. 提取模型基本信息
model_class_match = re.search(r'Model Class: (.*)', log_data)
model_path_match = re.search(r'Model Path: (.*)', log_data)

model_class = model_class_match.group(1) if model_class_match else "Unknown"
model_path = model_path_match.group(1) if model_path_match else "Unknown"

# 2. 正则表达式提取 Loss 数据
# 匹配模式：寻找 "500/500 ... - total_loss: ..." 这一行，因为它包含该 Epoch 的最终平均 Loss
pattern = re.compile(
    r'500/500 \[=+\] - .*? - total_loss: ([\d.]+) - weighted_task_loss: ([\d.]+) - decision_loss: ([-\d.]+) - paid_loss: ([\d.]+) - cost_loss: ([\d.]+) - val_total_loss: ([\d.]+) - val_weighted_task_loss: ([\d.]+) - val_paid_loss: ([\d.]+) - val_cost_loss: ([\d.]+) - val_decision_loss: ([-\d.]+)'
)

matches = pattern.findall(log_data)
data = []

for i, match in enumerate(matches, 1):
    data.append({
        # 'Epoch': i,
        # 'Model_Class': model_class,
        'Train_Total_Loss': float(match[0]),
        'Train_Weighted_Task_Loss': float(match[1]),
        'Train_Decision_Loss': float(match[2]),
        'Train_Paid_Loss': float(match[3]),
        'Train_Cost_Loss': float(match[4]),
        'Val_Total_Loss': float(match[5]),
        'Val_Weighted_Task_Loss': float(match[6]),
        'Val_Paid_Loss': float(match[7]),
        'Val_Cost_Loss': float(match[8]),
        'Val_Decision_Loss': float(match[9])
    })

# 3. 创建 DataFrame 并保存
df = pd.DataFrame(data)
print(df)
df.to_csv('log/training_log_alpha=2.0_res.csv', index=False)