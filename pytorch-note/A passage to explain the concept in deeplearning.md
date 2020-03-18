# PyTorch小本本:
CSDN：[https://blog.csdn.net/weixin_45709330/article/details/104883623](https://blog.csdn.net/weixin_45709330/article/details/104883623)
前言：以下内容为本人的拙见，出现理解错误或者描述不当的可以私信我改正😁
---
## ¿什么是神经网络？📚
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315184848559.png#pic_center)
神经网络包括三个layer：输入层、隐藏层、输出层
输入层：**用来将数据输入的**
隐藏层：处理数据的地方，处理之后再从输出层输出
输出层：输出结果

输入层和隐藏层之间的神经元利用两次连接的**权重进行数据传输**。 在权重矩阵中，利用随机数函数产生随机数给他们初始化
为了求概率值，在隐藏层中通过激活函数将输入进来的值转化为0到1之间的数
输入层向隐藏层传输就是通过输入矩阵和权重矩阵进行相乘，输入矩阵的每行乘权重矩阵的每列，即行跟列的点积，再传输到隐藏层。

> **什么是权重？**
> 权重就是用来连接一层之中每一个神经元和下一层的神经元
> 权重决定了对于神经元连接的强度。如果我们提高输入值很大程度上也开会影响输出。
> 权重接近零意味着更改此输入不会更改输出。 许多算法会自动将这些权重设置为零，以简化网络。
> Weights are used to connect the each neurons in one layer to the every neurons in the next layer.
Weight determines the strength of the connection of the neurons. If we increase the input then how much influence does it have on the output.
Weights near zero mean changing this input will not change the output. Many algorithms will automatically set those weights to zero in order to simplify the network.

>还是从比喻的角度来说吧，可以考虑我们学分绩的算法，权重就好比我们的学分，在神经网络中每一条网络下中的权重值可以通过影响每一条网络的输出值进而影响整个网络的输出，权重在不同的神经网络中还有不同的衍生，比如可以共享权值的卷积神经网络

**每一层神经网络值跟输入矩阵的一行和权重矩阵的一列进行点乘。**
再执行一次这个步骤就可以在输出层得到结果，也就是将上一部中隐藏层神经元中第0层得到的结果跟下一个在隐藏层和输出层之间的权重矩阵进行内积，最后得到的结果就是我们要的预测概率值(这个过程也就是正向传播算法，从输入层开始一层层往后计算，一直运算到结果层)
每层正向传播的神经网络可以解释为一个函数，函数的输入就是权重跟上层神经网络的输出。
**例**：
$$x=\begin{bmatrix}1 & 1 & -1 \\ 4 & 0 & 2 \\ 1 & 0 & 0\end{bmatrix}$$
$$weight=\begin{bmatrix}2 &-1 \\ 3 & -2 \\ 0 & 1\end{bmatrix}$$
$$layer_0=1*2+1*3+(-1)*0$$
$$weight_0=\begin{bmatrix}2 \\ 3 \\ 0 \end{bmatrix}$$
$$weight_1=\begin{bmatrix}-1 \\ -2 \\ 1 \end{bmatrix}$$
$$x^T=[x_1,x_2,...,x_n]$$
$$W=\begin{bmatrix}w_{11}&w_{12}&\cdots&w_{1j} \\ w_{21}&w_{22}&\cdots&w_{2j} \\ \vdots &&\cdots\\ w_{i1}&&\cdots&w_{ij}\\ \vdots&&\ddots \\w_{n1}&w_{n2}&\cdots&w_{nj}&\end{bmatrix}$$
通式可得：$$layer_n=\sum^n_{i=1}w_{ij}x_i$$
```python
x=[[1,1,-1],[4,0,2],[1,0,0]]
weight=[[2,-1],[3,-2],[0,1]]
layer0=1*2+1*3+(-1)*0 # the zero layer
weight0=weight[none][col=1] # the weight zero layer means the col 1 of list
weight1=weight[none][col=2] # the weight one layer means the col 2 of list

#sigmod - activation function
def nonlin(x):
	return 1/(1+np.exp(-x))
for j in rang(60000):
	#Feed forward through layer 0,1,2
	layer0=x
	layer1=nonlin.(np.dot(layer0,weight0))
	layer2=nonlin.(np.dot(layer1,weight1))
# calculate the error 
layer2_error = y - layer2 # y is the answer
```
输出之后跟原来正确的进行比较,发现每一个权重值$\omega$会对应一个错误率$E$（类似于二次函数）为了让这个错误率尽可能降低，所以我们要**用梯度下降来使错误率达到最低。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200316003915581.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200316004028801.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70#pic_center)

> **神经网络相当于是一个学习的部分，通过你给他的x,y来推断中间隐藏层的映射函数**
> **其实就是学习x，y之间对应关系的映射函数**

## 简易神经网络——感知机模型⚙

——在了解感知机之前我们先了解一下**M-P模型**
### M-P模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181228234402727.png#pic_center)
首先看到左边，左边是一列从$x_1$到$x_n$的参数，我们可以将这些参数看作类似生物神经元的树突接收到来自外界的信息，在传输进行时，每个参数都乘上一个对应权重值，权重从$\omega_{1j}$到$\omega_{nj}$，图中的大圆内就是用来判断输入的信息是否对输入的信息进行激活、输出的部分，在判断的时候及输出前对输入的信息使用$\Sigma$来进行求和，求和之后结果将作为输入送给函数$f$,$f$就是一个目标阈值的激活函数，只有满足目标阈值才能激活和输出。
$$y_j=f(\sum^n_{i=1}w_{ij}x_i-\theta_j)$$至此就搭建好了一个最简单的模型。

---
那么现在再看**单层感知机**，感知机就是一个能够进行二分类的**线性模型**（说白了就是把数据一分为二分成两类）只要数据能够**线性可分**，就能使用感知机模型不断地进行模型训练和参数优化。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200316175521559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70#pic_center)
<center>上图为单层感知机的示意图</center>

感知机的激活函数为$$f(x)=sign(x)=\begin{cases}+1 &x>0 \\ -1 & x<0\end{cases}\tag{1}$$
则感知机的数学表达式为:$$f(x)=sign(w\cdot x+b)\tag{2}$$
将$(2)$代入$(1)$得$$f(x)=sign(w\cdot x+b)=\begin{cases}+1 &w\cdot x+b>0 \\ -1 & w\cdot x+b<0\end{cases}$$
上述参数中：
$x:$输入的数据
输出的结果$\pm1$可以看作是被数据经过处理之后输出对应的标签。
这种感知机能够很容易处理**线性可分问题**，但不能处理**异或问题**（不能处理**非线性问题**）
当然，还有**多层感知机**，最大的区别就是在输入层和输出层之间加入了很多新的网络层次，通过自定义隐藏层的数量，使感知机具备了一种**后向传播能力**（理解为多次感知机模型能够对自我进行学习和优化）

### ¿什么是线性可分和线性不可分？
**线性不可分问题**简单来说就是你一个数据集不可以通过一个线性分类器（直线、平面）来实现分类。这样子的数据集在实际应用中是很常见的，**例如：人脸图像、文本文档**等。**我们不可以使用一个直线或者一个直面把上面图像中的两类数据很好的划分。这就是线性不可分。**
**线性可分**就是说可以用一个**线性函数把两类样本分开**，比如**二维空间中的直线、三维空间中的平面以及高维空间中的线性函数。**

所谓可分指可以没有误差地分开
## Pytorch框架
对于当时懵懂的我来说，图像的输入在哪？该如何将数据和网络结合，来训练一个自己的网络？
<font color=#ff0000>在pytorch中就只需要分三步，1.写好网络，2.编写数据的标签和路径索引，3.把数据送到网络。</font>

**1.1 PyTorch模型—网络架构**

1. **神经元零件进口**： 通过继承`class Net_name(nn.Module):`这个类，就能获取pytorch库中的零件啦；（点我看更多：[使用Module类来自定义模型](https://blog.csdn.net/qq_27825451/article/details/90705328) ， [使用Module类来自定义模型](https://blog.csdn.net/qq_27825451/article/details/90550890)）
2. **神经元零件预处理**： 像普通地写一个类一样，先要在`__init__(self)`中先初始化需要的“零件"(如 **conv、pooling、Linear、BatchNorm**等层)并让他们继承`Net_name`这个父类 ；其中可以通过`torch.nn.Sequetial`就是一个可以按照顺序一层层地封装初始化层的容器（[使用**Sequential**类来自定义顺序连接模型](https://blog.csdn.net/qq_27825451/article/details/90551513)）；下一步就可以在 `forward(self, x):`中用定义好的“组件”进行组装；
3. **神经元零件组装**： 通过编写 `def forward(self, x):`这个函数，$x$ 为**模型的输入**（就是你处理好的图像的入口啦），选取上一步中你处理好的**神经元零件**，在函数中按照你想构建的模型来拼凑，最终return出结果，而输入也会按照`forward`函数中的顺序通过神经网络，实现正向传播，最后输出结果。那么到此为止，你的模型就搭建完成啦！（[PyTorch之前向传播函数**forward**](https://blog.csdn.net/u011501388/article/details/84062483)）

**1.2 PyTorch模型—网络权值初始化**
	每个神经元零件一开始一般都会有初始化的参数的，当你的网络很深的时候，这些看似无关痛痒的初始参数就会对你的迭代过程以及最终结果有很大的影响。那么，这时候你就需要重新对这些网络层进行统一的规划安排，权值初始化方法就出来啦。**pytorch**中也提供了像是**Xavier**和**MSRA**方法来供你初始化，这里说下一般的初始化方法。

$\star$基础步骤：先设定什么层用什么初始化方法，实例化一个模型之后，执行该函数，即可完成初始化；
**按需定义初始化方法**，例如：

`if isinstance(m, nn.Conv2d):`
`n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels`
`m.weight.data.normal_(0, math.sqrt(2. / n))`
**初始化之后，在forward中就可以像积木一样，搭建自己整个神经网络。**

## Pytorch常用代码Note
<font color=#ff0000>torch的写法和numpy是相近的，但是要注意的是部分写法在反向传递的时候会有问题，自己写的时候就会发现啦，多写点</font>

- `torch.sum(input, dim, out=None)→ Tensor` #与python中的sum一样
$input (Tensor)$ – 输入张量
$dim (int)$ – 缩减的维度，开始的时候不理解为什么英文文档里面会说这是缩减的维度，对于高维度的数组如$(Q\times W\times E\times R)$，对$dim=1$进行$sum$，那么其得到的维度就是$Q\times E\times R$，保留最高维度$Q$的形状，对$W$维度对应最小的元素进行求和，所以$W$维度就会消失，其他的函数的维度处理也是这样理解。
`out (Tensor, optional)` – 结果张量
`print(x.sum(0))`#对一维的数求和，按列求和
`print(x.sum(1))`#对二维求和按行求和
`print(x.sum(2))`#将最小单位的数组元素相加即可
- `new_features = super(_DenseLayer, self).forward(x)`
最后在官方论坛上得到结果，含义是将调用所有`add_module`方法添加到`sequence`的模块的`forward`函数。
- `torch.where(condition, x, y) → Tensor`对于$x$而言，如果其中的每个元素都满足condition，就返回$x$的值；如果不满足condition，就将y对应位置的元素或者$y$的值
- `torch.narrow(input, dimension, start, length)` 张量剪裁
permute把张量变换成不同维度，view相当于reshape，将元素按照行的顺序放置在新的不同大小的张量当中
- `torch.cat(tensors, dim=0, out=None) → Tensor`将张量按照维度进行衔接
- `torch.gesv(B, A, out=None) -> (Tensor, Tensor)`是解线性方程$AX=B$后得到的解
(1) `torch.unsqueeze(input, dim, out=None) → Tensor` 在指定位置增加一个一维的维度
dim (int) – the index at which to insert the singleton dimension
(2) `torch.squeeze(input, dim, out=None) → Tensor` 在指定位置减去一个一维的维度，默认()就是把所有shape中为1的维度去掉
- `detach()`就是取出一个该个tensor，并且它不会再参与梯度下降
- `torch.nn`
`nn.Sequential`一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。

`torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')`将其归一化到一个$（-1,1）$的二维平面上，outpuy_{x,y} 的像素值与`input_{x0,y0}` 的像素值一致， 矩阵索引通过grid矩阵保存。 `grid_{x,y}=(x0,y0)`
参考网站 [https://blog.csdn.net/houdong1992/article/details/88122682](https://blog.csdn.net/houdong1992/article/details/88122682)
英文手册原文 [https://pytorch.org/docs/stable/nn.html?highlight=grid_sample#torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/nn.html?highlight=grid_sample#torch.nn.functional.grid_sample)
[transforms的使用方法](https://blog.csdn.net/weixin_38533896/article/details/86028509#1transformsRandomCrop_38)
