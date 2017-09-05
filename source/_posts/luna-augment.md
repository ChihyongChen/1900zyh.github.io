---
title: Practice | LUNA: 数据增强
date: 2017-09-04 21:20:55
tags:
---

经过上一篇blog的介绍，我们只是从原始的mhd数据进行ROI处理，并统一分辨率。经过这样一系列的操作，我们得到的是每个病例不同大小的数据，而且数据集很小（1800+个结节）。如何经过裁剪、数据增强等一系列操作将数据送入检测网络是本篇blog要讨论的事情。

这段代码对数据进行划分、预处理，然后送入网络，并得到实验结果。我们需要讨论的三个步骤：split_comber, DataBowl3Detector, test_detect.
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

SplitComb是一个类，类的主要参数有：side_len=144, max_stride=16, stride=4, margin=32, pad_value=170
```python
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
```

DataBowl3Detector是继承torch的一个类Dataset。关于torch的教程，可以参考[这里](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)。
```python
class DataBowl3Detector(Dataset):
    def __init__(self, split, config, phase = 'train',split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']       
        self.stride = config['stride']       
        sizelim = config['sizelim']/config['reso']
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        data_dir = config['datadir']
        self.pad_value = config['pad_value']
        
        self.split_comber = split_comber
        idcs = split
        if phase!='test':
            idcs = [f for f in idcs if f not in self.blacklist]

        self.channel = config['chanel']
        if self.channel==2:
            self.filenames = [os.path.join(data_dir, '%s_merge.npy' % idx) for idx in idcs]
        elif self.channel ==1:
            if 'cleanimg' in config and  config['cleanimg']:
                self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
            else:
                self.filenames = [os.path.join(data_dir, '%s_img.npy' % idx) for idx in idcs]
        self.kagglenames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])>20]
        self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])<20]
        
        labels = []
        
        for idx in idcs:
            if config['luna_raw'] ==True:
                try:
                    l = np.load(os.path.join(data_dir, '%s_label_raw.npy' % idx))
                except:
                    l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
            else:
                l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
            labels.append(l)

        self.sample_bboxes = labels
        if self.phase!='test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for t in l:
                        if t[3]>sizelim:
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3]>sizelim2:
                            self.bboxes+=[[np.concatenate([[i],t])]]*2
                        if t[3]>sizelim3:
                            self.bboxes+=[[np.concatenate([[i],t])]]*4
            self.bboxes = np.concatenate(self.bboxes,axis = 0)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

```





# 划分数据: split_comber


```python
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)
        splits = []
        _, z, h, w = data.shape
        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw
        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin
                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)
        splits = np.concatenate(splits, 0)
        return splits,nzhw
```



```python
    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride
        splits = []
        for i in range(len(output)):
            splits.append(output[i])
        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)
        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1
        return output 
```





# 数据切割




# 数据增强




# 训练网络




# 实验结果



