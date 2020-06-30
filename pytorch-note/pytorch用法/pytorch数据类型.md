# Pytorch中的数据类型



## 数据类型比较

|   python    |      pytorch      |
| :---------: | :---------------: |
|     Int     |     IntTensor     |
|    float    |    FloatTensor    |
|  Int array  |  IntTensor array  |
| Float array | FloatTensor array |
|   String    |         -         |



## 位置不同的数据类型（CPU和GPU）

| 数据类型      | CPU Tensor         | GPU Tensor              |
| ------------- | ------------------ | ----------------------- |
| torch.float32 | torch.FloatTensor  | torch.cuda.FloatTensor  |
| torch.float64 | torch.DoubleTensor | torch.cuda.DoubleTensor |
| torch.uint8   | torch.ByteTensor   | torch.cuda.ByteTensor   |
| torch.int32   | torch.IntTensor    | torch.cuda.IntTensor    |
| torch.int64   | torch.LongTensor   | torch.cuda.LongTensor   |



## 如何检查数据类型

```python
a = torch.randn(2,3)
```



- 方法1

```python
a.type()
# 输出 torch.FloatTensor
```

- 方法2

```python
type(a)
# 输出 torch.Tensor
```

- 方法3

```python
isinstance(a, torch.FloalTensor)
# 输出 True
```

- 方法4

```python
a.dtype
# 输出 torch.FloatTensor
```



## 将数据搬到GPU上

```python
a = torch.randn(3,3)
a = a.cuda()
print(a.dtype)
# 输出 torch.cuda.DoubleTensor
```





# 查看张量大小

```python
a = torch.randn(2,3)
```

- 方法1

```python
a.shape
# 输出 (2,3)
```

- 方法2

```python
a.size()
# 输出 torch.size([2,3])
```

- 方法3

```python
a.numel()
# 输出 6
```

- 方法4

```python
a.dim()
# 输出 2
```



# 数据类型转换

```python
a = np.array([2,3.3])
```



- numpy转torch

```python
torch.from_numpy(a)

# 输出 tensor([2.000,3.3000], dtype=torch.float64)
```



- list转torch

```python
torch.tensor([2,3.3])
# 输出 tensor([2.000,3.3000])
```

> 注: torch.Tensor跟torch.FloatTensor类似，接受的参数是张量的大小或者list，而torch.tensor接受的参数是list



- 其他数据类型转torch

```python
tensor = transforms.ToTensor()(array)
```



- torch转PIL

```python
tensor = transforms.ToPILImage()
```



- torch转numpy

```python
array = tensor.numpy()
```



<font color="red">***注意：Tensor的形状是[C,H,W]，而cv2，plt，PIL形状都是[H,W,C]**</font>

## 创建Tensor

- torch.tensor()

这个可以生成一个张量，传递的参数是list

- torch.Tensor()

这个可以生成一个张量，传递的参数可以是==list或大小==（这个一般按照pytorch默认的数据类型来生成）

- torch.FloatTensor()

这个可以生成一个浮点类型的张量，其中传递的参数可以是==列表==，也可以是==维度值==

- torch.IntTensor()

这个可以生成一个整型类型的张量，其中传递的参数可以是==列表==，也可以是==维度值==

- torch.rand()

这个函数可以生成数据为==浮点类型且维度指定的==随机张量，与numpy中的`numpy.rand()`类似。生成的数范围在$[0,1]$之间，传递的参数是==张量大小==

- torch.randn()

这个函数可以生成数据为==浮点类型且维度指定==的随机张量，与numpy中的`numpy.rand()`类似。生成的数取值满足均值为0、方差为1的==正态分布==

- torch.randint()

这个函数可以生成数据为==整型类型且维度指定的==随机张量，与numpy中的`numpy.rand()`类似。生成的数范围在$[0,1]$之间，传递的参数是==最小值，最大值，张量大小==

- torch.range()

这个函数可以生成数据为==浮点类型且定义范围==的张量，传递3个参数，分别为范围的==起始值，范围的结束值，每个数据之间的间隔==

- torch.arange()

这个函数可以生成有一定规律的张量，传递3个参数，起始值，结束值，间隔

- torch.full()

这个函数可以把所有的元素==赋值成相同的值==，传递两个参数，list（张量大小）和要赋的值

- torch.normal()

这个函数可以根据张量的==均值和方差==生成张量

- torch.zeros()

这个可以生成数据类型为==浮点且维度指定==的张量，且所有元素都是0，即可以生成一个==零张量==。

- torch.ones()

这个可以生成数据类型为==浮点且维度指定==的张量，且所有元素都是1。

- torch.eye()

这个可以生成一个数据类型为==浮点且主对角线为1的==张量，即生成一个==单位张量==

- torch.linspace()

这个可以生成按==一定比例划分连续==的一个张量，传递的参数有起始值，结束值，分成几份。（其实就是按照线性划分）

- torch.logspace()

这个是按照==对数空间==上进行划分，与上面一个相同

- torch.randperm()

这个是可以把张量进行打乱，进行一个shuffle的操作。传递的参数是==维度==，要打乱第几维度