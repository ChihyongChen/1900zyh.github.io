---
title: Practice | LUNA: 数据增强
date: 2017-09-04 21:20:55
tags:
---

经过上一篇blog的介绍，我们只是从原始的mhd数据进行ROI处理，并统一分辨率。经过这样一系列的操作，我们得到的是每个病例不同大小的数据，而且数据集很小（1800+个结节）。如何经过裁剪、数据增强等一系列操作将数据送入检测网络是本篇blog要讨论的事情。

这段代码对数据进行划分、预处理，然后送入网络，并得到实验结果。我们需要讨论的三个步骤：
```python
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)

    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])
```


# 划分数据





# 数据切割




# 数据增强




# 训练网络




# 实验结果



