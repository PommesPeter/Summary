# VOC和COCO数据集的区别



### voc数据集

VOC数据集的标签均为`xml`文件格式，其中包括了boundingbox的位置信息，里面的**xmin、ymin、xmax、ymax**代表了**方形矩形框**的位置，分别是**纵向和横向最大和最小值**，即可确定一个矩形框的位置。

VOC数据集中不是所有的图片都能够用来分割，因为分割需要在物体轮廓边缘填充一定的颜色。

- 文件的结构一般为

```
-JPEGImage # 存储所有的图片像信息，包括训练图片和测试图片
-Annotations # 存储xml格式的标签，每个xml文件对应了一张图片
-ImageSets
--Main  # 存放图像物体识别的数据，也就是训练图片和测试图片的名称（不包括后缀）
```



### coco数据集

coco数据集的标签为json格式，包括了boundingbox的精确坐标，以Segmentation像素的位置。所以coco数据集包括了，目标检测、关键点检测、stuff没有固定形状的物体检测、全景分隔（things和stuff全被分割）、“看图说话”（一个图片中的场景描述）等

其中coco就包含了三种标签：

**object instances（目标实例）**

**object keypoints（目标上的关键点）**

**image captions（看图说话）**

>  [知乎上对于coco数据集结构的介绍](https://zhuanlan.zhihu.com/p/29393415)

#### 格式

json文件中一般是这样的：

```
{"segmentation":[[392.87,275.77, 402.24, 284.2, 382.54, 342.36, 375.99, 356.43, 372.23, 357.37, 372.23,397.7, 383.48, 419.27,407.87, 439.91, 427.57, 389.25, 447.26, 346.11, 447.26,328.29, 468.84, 290.77,472.59, 266.38], [429.44,465.23, 453.83, 473.67, 636.73,474.61, 636.73, 392.07, 571.07, 364.88, 546.69,363.0]], "area":28458.996150000003, "iscrowd": 0,"image_id": 503837, "bbox":[372.23, 266.38, 264.5,208.23], "category_id":4, "id": 151109}
```

- 文件结构也比较简单

  ```
  -annotations
  -train
  -val
  ```

  



### things和stuff的区别

thing和stuff的区别：

2018CVPR论文**《COCO-Stuff: Thing and Stuff Classes in Context》**里是这么写的：

> Defining things and stuff. The literature provides definitions for several aspects of stuff and things, including:(1) Shape: Things have characteristic shapes (car, cat,phone), whereas stuff is amorphous (sky, grass, water)[21, 59, 28, 51, 55, 39, 17, 14]. (2) Size: Things occur at characteristic sizes with little variance, whereas stuff regions are highly variable in size [21, 2, 27]. (3) Parts: Thing classes have identifiable parts [56, 19], whereas stuff classes do not (e.g. a piece of grass is still grass, but a wheel is not a car). (4) Instances: Stuff classes are typically not countable [2] and have no clearly defined instances [14, 25, 53]. (5) Texture: Stuff classes are typically highly textured [21, 27, 51, 14]. Finally, a few classes can be interpreted as both stuff and things, depending on the image conditions (e.g. a large number of people is sometimes considered a crowd).

通过paper中的描述，我们可以理解为，things就是我们所需要识别的具体的一些物体，比如车、猫、狗。而stuff则就是相当于背景的事物，比如天空、草、水流等物体。

