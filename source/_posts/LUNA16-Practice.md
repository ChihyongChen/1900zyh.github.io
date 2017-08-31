---
title: LUNA16_Practice
date: 2017-08-14 21:21:51
tags:
---


本篇研究的是[Data Science Bowl 2017 TOP1](https://github.com/lfz/DSB2017)的代码。
数据类型是mhd格式。
在mhd格式中，可以根据HU值来区别空气、组织器官和骨头等等，从而将肺部组织提取出来。
关于HU值的介绍可以看这篇[常见医疗扫描图像处理步骤](https://shartoo.github.io/medical_image_process/)。
接下来，我将详细分析这个TOP1的代码。
<!--more-->


# 预处理：提取肺部组织

## 每张slice单独分析

首先是预处理部分，将肺部组织提取出来，去掉其他没有用的噪声是必要的。主要步骤有如下：

```python
def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)    
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), 
                sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
    return bw
```

这段代码是对一个病人的每一张slice进行分析。接下来细说这个函数的每个步骤。
首先计算距离。在一张slice中，以slice中心点为原点，计算这个slice中各个地方到中心点的几何距离。将距离大于等于image_size/2的设为nan。
```python
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    # d is the distane for all pixel to the center
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
```
对于每一张slice，如果左上角$10 \times 10$的区域里，HU值都是一样的即都是空气或水，就用nan_mask将整张图的非中心区域去掉，否则就保留原值。然后再用用size = 1 pixel的高斯滤波器先进行滤波。最后，将$ HU >= -600 $的过滤，因为肺部组织的HU值是-500.
```python
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply.image[i].astype('float32'), nan_mask), 
                sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
```

skimage.measure.label函数是用来实现将上面求得的bool值进行连通区域标记。regionprops函数对每一个连通区域进行属性获取和操作，比如计算面积、外接矩形、凸包面积等等。计算结果返回为所有连通区域的属性列表，列表长度为连通区域个数。如果连通区域面积大于area_th（这里设为$30mm^2$）,并且连通区域的离心率要小于阈值(eccentricity < 0.99),就认为这个区域是ROI。 此时就对每张slice生成了一个mask。

```python
    label = measure.label(current_bw)
    properties = measure.regionprops(label)
    valid_label = set()
    for prop in properties:
        if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
            valid_label.add(prop.label)
    current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    bw[i] = current_bw
```



## 所有slices立体分析

首先这部分代码整体是这样的。
```python
    bw = binarize_per_slice(img_array, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
```

```python
def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    return bw, len(valid_label)
```
接下来详细分析每一个步骤。

在某一些例子中，需要把最上面的几张slice去掉。cut_num每次以cut_step（这里设为2）的速度递增，也就是每两张判断一次的意思。
```python
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
```` 

2.3 bg_label是第一张slice的四个角以及第一条和最后一条的中点处、除去被cut掉的slice后的最后一张的四个角以及第一条和最后一条的中点处的label。将这些label都设为0.这里将这些部分的区域都当成了背景，设为0，其实也就是设成label[0,0,0]。
```python
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], 
                    label[0, -1, 0], label[0, -1, -1],
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], 
                    label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1],
                    label[0, 0, mid], label[0, -1, mid], 
                    label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
```


2.4 这里的vol_limit是[0.68L, 7.5L]。上一步将所有slice的背景部分都设成了同样的label值，这里算出的prop.area就是立体的，乘以spacing的三个数相乘就得到了物理意义上的体积。这里设定体积小于0.68L，大于7.5L的都去掉。这里是认为大于7.5L的是背景，小于0.68L的是杂质之类的。
```python
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
``` 


2.5 和1.1节中的单独对每张slice进行分析时候一样，计算一张map，map上的每个位置是该位置到slice中心位置的几何距离，不同的是此时的几何距离不是简单的图像上的距离，而是物理世界里的几何距离。
```python
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
```

5. vols表示目前3D slices中所有的有效的label。对于每一种label， 计算每张slice上这种label的有效面积slice_area，min_distance记录的是每张slice上这种label的区域距离该slice中心最近的距离。我们认为这个病人的所有面积大于阈值（area_th：$6e3 mm^2$）是有效slice，而这些有效slice距离各自中心点的平均距离如果小于阈值（dist_th： 62mm）则认为这个label是属于肺部组织的label。
```python
    vols = measure.regionprops(label)
    valid_label = set()
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
```

6. 
```python
    if cut_num > 0:
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    return bw, len(valid_label)
```

3. fill hole 的意思是。。。。这段处理的代码如下
```python
def fill_hole(bw):
    label = measure.label(~bw)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], 
                    label[0, -1, 0], label[0, -1, -1],
                    label[-1, 0, 0], label[-1, 0, -1], 
                    label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)    
    return bw
```
```python
    bw = binarize_per_slice(img_array, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
```


## 单独生成左右肺mask
```python
def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label
        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice
        return bw
    
    print("two lung only...")
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        print(iter_count, max_iter)
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
    else:
        print("***************not found***************")
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
    print("fill_2d_hole ing")
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2
    return bw1, bw2, bw
```


## 统一的分辨率



