---
title: Practiec | LUNA16：数据预处理
date: 2017-08-14 21:21:51
tags: CT, Research
---


本篇研究的是[Data Science Bowl 2017 TOP1](https://github.com/lfz/DSB2017)的代码。
数据类型是mhd格式。
在mhd格式中，可以根据HU值来区别空气、组织器官和骨头等等，从而将肺部组织提取出来。
关于HU值的介绍可以看这篇[常见医疗扫描图像处理步骤](https://shartoo.github.io/medical_image_process/)。
接下来，我对这些代码进行了轻微的改动，然后分析每一步的作用。 
<!--more-->


# 预处理：提取肺部组织
数据预处理部分我感觉是涉及到了很多的医学知识。
主要过程是：滤波、二值化、联通区域分析、3D联通区域分析、距离中心点距离分析。
整个过程有点迭代的意思，还挺琐碎复杂的。

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

对于每一张slice，如果左上角$10 \times 10$的区域里，HU值都是一样的即都是空气或水，就将整张图的非中心区域设为nan，否则就保留原值。然后再用用size = 1 pixel的高斯滤波器先进行滤波。然后用$ HU >= -600 $将图像二值化，因为肺部组织的HU值是-500.
```python
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply.image[i].astype('float32'), nan_mask), 
                sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
```

经过上述的处理后，最后进行连通区域分析移除background成分。skimage.measure.label函数是用来实现将上面求得的bool值进行连通区域标记。regionprops函数对每一个连通区域进行属性获取和操作，比如计算面积、外接矩形、凸包面积等等。计算结果返回为所有连通区域的属性列表，列表长度为连通区域个数。如果连通区域面积大于area_th（这里设为$30mm^2$）,并且连通区域的离心率要小于阈值(eccentricity < 0.99),就认为这个区域是ROI。 此时就对每张slice生成了一个mask。
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

下面两张图分别是处理前后用matplotlib画出来的。
处理前：
![处理前](load_NO1_slice60.png)
处理后：
![处理后](binarized_NO1_slice60.png)
可以看到处理后，边角的背景、包裹肺部的亮色胸腔部分都被去除了。
也就是说，这一步主要就是去边角、肺部组织附近的脂肪、水、肾等background。


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
```

bg_label是第一张slice的四个角以及第一条和最后一条的中点处、除去被cut掉的slice后的最后一张的四个角以及第一条和最后一条的中点处的label。将这些label都设为0.这里将这些部分的区域都当成了背景，设为0，其实也就是设成label[0,0,0]。
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

这里的vol_limit是[0.68L, 7.5L]。上一步将所有slice的背景部分都设成了同样的label值，这里算出的prop.area就是立体的，乘以spacing的三个数相乘就得到了物理意义上的体积。这里设定体积小于0.68L，大于7.5L的都去掉。这里是认为大于7.5L的是背景，小于0.68L的是杂质之类的。
```python
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
```

与之前每张slice单独进行分析时候一样，计算一张map，map上的每个位置是该位置到slice中心位置的几何距离，不同的是此时的几何距离不是简单的图像上的距离，而是物理世界里的几何距离。
```python
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
```

vols表示目前3D slices中所有的有效的label。对于每一种label， 计算每张slice上这种label的有效面积slice_area，min_distance记录的是每张slice上这种label的区域距离该slice中心最近的距离。我们认为这个病人的所有面积大于阈值（area_th：$6e3 mm^2$）是有效slice，而这些有效slice距离各自中心点的平均距离如果小于阈值（dist_th： 62mm）则认为这个label是属于肺部组织的label。
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

最后要将第一步去除掉的几张slice还原回来。bw1是处理过但保留原cut_num张slice的图像，bw2是不还原并且进行过膨胀处理的mask。最终的mask：bw3，是bw1和bw2的交集。valid_l3中的元素表示的就是bw中有并且bw中也有的。
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

以上的这些步骤在某种情况下会迭代进行。这是因为，在某些case中，肺部组织会和外部空间连在一起，这就导致这些区域是联通的、超过阈值的，会被当成背景，此时valid_label可能是空的，因此就需要把top的几张slice先去除。这里使用cut_num=2的速度来进行筛选。
```python
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
```

如下图就是一个处理后的示意图，将一些仪器背景形成的联通区域去掉了。

![处理后](3danalysis_NO1_slice60.png)

从上面这张图我们会发现，肺部中间有些洞被空挖掉了，但实际上我们希望提取ROI是整个肺部组织，而不是挖洞的。因此，我们再次进行联通区域分析，并且只将label和边角的一样的部分移除，其余的保留。这段处理的代码如下：
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

这一步结束后得到的结果：
![处理后](fillhole_NO1_slice60.png)
可以看到有一些洞确实是被填上了。二维的角度看好像没什么变化，但实际上三维角度上看剩下的这些洞很可能是和外部空间相连接的，因此没有被填上。

## 单独生成左右肺mask

这部分代码主要是，不断进行腐蚀操作直到最大的两个区域（左肺和右肺）有同样的体积。在腐蚀膨胀的过程中，分别为两片肺生成mask。
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
以下对这部分的代码进行一一分析。

按照代码执行的逻辑顺序开始分析。这里有两个参数，一个是max_iter，设定的是迭代寻找肺部组织的次数，程序中用的是22。一个是max_ratio，最大面积不能超过第二面积的max_ratio(这里设为4.8)倍。找到符合面积要求的联通区域则停止迭代，否则就继续腐蚀操作直到找到符合要求的联通区域。这里可以看到，bw1是最大面积的label，bw2是第二大面积的label，从之后的代码也可以看出，这里的bw1和bw2其实指的就是左右肺。注意这里的联通区域是三维的。
```python
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
```
bw1:
![bw1](left_max_area.png)
bw2:
![bw2](right_second_area.png)

上述步骤找到左右肺后继续下一步操作，如果没有找到左右肺就直接将原本的bw作为bw1,而bw2为空，二者并集为最终的mask。distance_transform_edt的意思是非零点到背景点（零值点）的最近距离。这一步实际上是将左右肺继续做明确的分割，然后再进行主要成分提取。最后还是进行一些肺部内部区域的填充.
```python  
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

上述中的extract_main函数分析如下。首先还是进行联通区域分析，选择占当前slice 95% 面积的区域，并计算他们的凸包convex_image的并集。通过这种方式可以将肺部的主要面积给截取出来。但最后返回的bw是面积最大的那一个。
```python
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
```

fill_2d_hole主要是利用联通区域的filled_image进行处理。
```python
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
```
该过程处理后的一个最终结果：
bw1:
![bw1](bw1_after_extract_masks.png)
bw2:
![bw2](bw2_after_extract_masks.png)
bw:
![bw](bw_after_extract_masks.png)


# 预处理：Apply Mask

这一步首先先计算出物理世界中的ROI的外边框。然后将mask再分别向两边膨胀5个物理坐标点，使得mask周围的一点空间可以被包进来。
```python
    resolution = np.array([1,1,1])
    Mask = m1+m2
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')
````

。如果凸包mask相比原mask增加的区域超过了50%，则用原mask继续计算，否则就用凸包图像convex hull来代替目前计算的mask。最后返回腐蚀操作的dilatedMask。
```python
def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        # ascontiguousarray函数是将数组以连续数组的形式返回
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            # mask2是当前mask的凸包图像
            mask2 = convex_hull_image(mask1)
            # 如果凸包图像比原mask大了50%
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
        # connectity为1
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask
```

这段代码就是利用process_mask操作来对mask进行凸包处理。
```python
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask - Mask
```
以下这张图就分别是dilated_mask, mask,extramask。
![dilated_mask_extra](dilated_mask_extra.png)

simpleitk读取mhd的image_array数值是HU的数值，在送进网络进行判断前，我们需要将其转化成[0,255]的灰度值。HU有效值在[-1200,600]之间。整个映射变化只是一个简单的线性映射。
```python
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg
```

由于不同的病人之间的spacing（单位：mpp,毫米/像素）是不同的，这意味着物理世界中同样大小的面积在图像中表示不一样。所以我们认为应该要统一spacing。
```python
def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
```

这部分代码就是将HU值转化成灰度值（从[-1200,600]到[0，255]），并应用上上一步求的mask，把非uROI的部分用170填充。除此之外，由腐蚀操作多出来的面积如果灰度值大于210，被认为是骨头的部分，也被填充成170.通过resample统一到一样的分辨率后，再用计算到的extendbox将ROI截取出来, 这里通过extendbox截取的时候是x\y\z三个方向上都有进行截取。
```python
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
```
灰度值转化且统一分辨率后裁剪前：
![origin](origin.png)
裁剪后：
![cut](cut.jpg)
由于经过裁剪，总的slice数量不一样了，因此这两张图不是一一对应的。


# 预处理：处理annatations
上面一个步骤完成后，图像也就处理好了，但还需要把这些数据的label给对应到新的位置上来。
这部分函数是将世界坐标（ground truth的坐标）映射到图像的voxel坐标系上。
```python
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord
```

由于slice是经过缩放、裁剪的，因此labe也需要经过缩放裁剪。
```python
    this_annos = np.copy(annos[annos[:,0]==case_name])
    label = []
    if len(this_annos) > 0:
        for c in this_annos:
            pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
            label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
    label = np.array(label)
    
    if len(label) == 0:
        label2 = np.array([0,0,0,0])
    else:
        label2 = np.copy(label)
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    print(label2)
    np.save(os.path.join(ans_path,case_name+'_label.npy'),label2)
```
裁剪前的结节位置：
![origin_label](origin_label.png)
裁剪后的结节位置：
![cut_label](cut_label.png)



预处理部分就先介绍到这里，下一步将分析如何将数据送入网络。（这其中就涉及到了数据增强等操作啦。）