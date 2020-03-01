# Local Color Correction（局部颜色校正）
> 以下大部分引用自：[https://www.yanxishe.com/columnDetail/16833](https://www.yanxishe.com/columnDetail/16833)
## 前言
这个是出自一篇论文里面的算法。论文地址为：[http://www.ipol.im/pub/art/2011/gl_lcc/](http://www.ipol.im/pub/art/2011/gl_lcc/) 。IPOL 是一个非常好的学习数字图像处理的网站，上面的论文都是提供配套源码的，如果平时在数字图像处理方面想找一些 Idea可以上这个网站。

## 什么是掩膜图像？

> 下面引自
> 作者：bitcarmanlee
> 链接：[https://blog.csdn.net/bitcarmanlee/article/details/79132017](https://blog.csdn.net/bitcarmanlee/article/details/79132017)

在图像处理中，经常会碰到掩膜(Mask)这个词。那么这个词到底是什么意思呢？下面来简单解释一下。

#### 1.什么是掩膜
在半导体制造中，许多芯片工艺步骤采用光刻技术，用于这些步骤的图形“底片”称为掩膜（也称作“掩模”）
其作用是：在硅片上选定的区域中对一个不透明的图形模板遮盖，继而下面的腐蚀或扩散将只影响选定的区域以外的区域。
图像掩膜与其类似，用选定的图像、图形或物体，对处理的图像（全部或局部）进行遮挡，来控制图像处理的区域或处理过程。

#### 2.掩膜的用法
2.1 提取感兴趣区：用预先制作的感兴趣区掩膜与待处理图像相乘，得到感兴趣区图像，感兴趣区内图像值保持不变，而区外图像值都为0；
2.2 屏蔽作用：用掩膜对图像上某些区域作屏蔽，使其不参加处理或不参加处理参数的计算，或仅对屏蔽区作处理或统计；
2.3 结构特征提取：用相似性变量或图像匹配方法检测和提取图像中与掩膜相似的结构特征；
2.4 特殊形状图像的制作。

#### 3.掩膜运算的一个小实例
以图和掩膜的与运算为例：
原图中的每个像素和掩膜中的每个对应像素进行与运算。比如1 & 1 = 1；1 & 0 = 0；
比如一个3 * 3的图像与3 * 3的掩膜进行运算，得到的结果图像就是：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200221024243311.png)
#### 4.小结
1.图像中，各种位运算，比如与、或、非运算与普通的位运算类似。
2.如果用一句话总结，掩膜就是两幅图像之间进行的各种位运算操作。

## 算法原理
首先对于太亮和太暗的图像，我们可以使用 Gamma 校正和直方图均衡化来提高对比度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022023362294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70)
上图分别代表**较暗图像**，**Gamma 系数为 0.5 的 Gamma 校正**，**直方图均衡化**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200220233630324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70)
上图分别代表较亮的**原始图像**，**Gamma 系数为 2.5 的 Gamma 校正**，**直方图均衡化**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200220233640200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70)
上图分别代表**原始图像**，**Gamma 系数为 0.5,2.5,0.75,1.5 的 Gamma 校正图像**

使用 Gamma 校正后可以**提高图像的动态范围**，实际上作者讲这么多实际是要说，如果当图像既有**较亮又有较暗**的
区域时，如果仅仅使用一个 Gamma 矫正输出的图像效果反而会变差，这是因为 **Gamma 矫正是全局**的方法，某一部分相近的像素将被映射到相同的灰度值，并没有考虑待到像素邻域的信息。对于普通的过亮和过暗的图像，当图像的平均灰

度大于 127.5 使用$\gamma>1$，**对图像的亮度进行抑制**；当图像的灰度信息均值小于 127.5 时使用$\gamma<1$**对图像亮度进行增强**。这里我们假设图像用无符号 8bit 表示，那么$γ=2^{ \frac{u−127.5}{127.5} }$ 。在既有**较暗又有较亮的区域的图像中，全局 Gamma 失效**，这时候作者就提出了**利用图像邻域**的信息，进行 Gamma 矫正。对**较暗的区域进行增加亮度，对较亮的区域降低亮度**。局部颜色校正的方法可以根据**邻域内像素的灰度值情况**，把统一输入像素值，映射成不同水平的像素灰度值。

## 算法步骤

 - 根据输入图像计算出掩膜图像
 - 结合输入图像和掩模图像计算出最终结果

掩膜图像一般根据**彩色图像各个通道的图像灰度值获得**。假设 RGB 图像各个通道的像素灰度值为 R，G，B，则掩膜图像可以表示为$I=\frac{(R+G+B)}{3}$，之后对掩膜图像进行**高斯滤波**:$M(x,y)=(Gaussian∗(255−I))(x,y)$，高斯滤波时，选取**较大值进行滤波**，以保证**对比度不会沿着边缘方向过度减小**。上述的输出结果表明：**图像哪部分需要提亮，哪部分需要减暗。**
最后输出图像为：$Output(x,y)=255(\frac{Input(x,y)}{255})^{2^{\frac{128-M(x,y)}{128}}}$
如果掩膜图像大于 128，将得到一个大于 1 的指数，并对图像该点的亮度移植，反之增加亮度。如果等于 128，则不改变该像素点亮度。

## C++代码

```cpp
Mat LCC(const Mat &src){
    int rows = src.rows;
    int cols = src.cols;
    int **I;
    I = new int *[rows];
    for(int i = 0; i < rows; i++){
        I[i] = new int [cols];
    }
    int **inv_I;
    inv_I = new int *[rows];
    for(int i = 0; i < rows; i++){
        inv_I[i] = new int [cols];
    }
    Mat Mast(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i++){
        uchar *data = Mast.ptr<uchar>(i);
        for(int j = 0; j < cols; j++){
            I[i][j] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[1]) / 3.0;
            inv_I[i][j] = 255;
            *data = inv_I[i][j] - I[i][j];
            data++;
        }
    }
    GaussianBlur(Mast, Mast, Size(41, 41), BORDER_DEFAULT);
    Mat dst(rows, cols, CV_8UC3);
    for(int i = 0; i < rows; i++){
        uchar *data = Mast.ptr<uchar>(i);
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < 3; k++){
                float Exp = pow(2, (128 - data[j]) / 128.0);
                int value = int(255 * pow(src.at<Vec3b>(i, j)[k] / 255.0, Exp));
                dst.at<Vec3b>(i, j)[k] = value;
            }
        }
    }
    return dst;
}
```

代码分析：

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
//这个算法的原理就是通过遍历图像上每一个像素点，通过对掩膜图像进行高斯滤波处理来进行选择图像哪部分需要提亮，哪部分需要减暗。
//高斯滤波时，选取较大值进行滤波，以保证对比度不会沿着边缘方向过度减小。上述的输出结果表明：图像哪部分需要提亮，哪部分需要减暗。
//如果掩膜图像大于 128，将得到一个大于 1 的指数，并对图像该点的亮度移植，反之增加亮度。如果等于 128，则不改变该像素点亮度。
Mat LCC(const Mat& src)
{
    int rows = src.rows;//获取图像的横向长度
    int cols = src.cols;//获取图像的纵向长度

    //申请一个动态整型数组，数组的长度为[]中的值，存放横向方向的像素 
    //这里是声明存放掩膜图像的数组
    int** I = new int* [rows];
    
    for (int i = 0; i < rows; i++) {
        //创建一个int型数组，数组大小是在[]中指定
        //在这里是在一维数组I中每一个元素再定义一个数组存放纵向方向的像素
        //相当于定义了一个二维数组
        I[i] = new int[cols];
    }
    //以下部分也是创建一个inv_I的二维数组且是动态整型数组
    int** inv_I;
    inv_I = new int* [rows];
    for (int i = 0; i < rows; i++) {
        inv_I[i] = new int[cols];
    }

    Mat Mast(rows, cols, CV_8UC1);//掩膜图像处理
    for (int i = 0; i < rows; i++) {
        uchar* data = Mast.ptr<uchar>(i);//？？？
        for (int j = 0; j < cols; j++) {
            //获取输入图像的R,G,B三个通道颜色像素的灰度值以得到掩膜图像
            //src.at<Vec3b>(i, j)[0] src.at<Vec3b>(i, j)[1] src.at<Vec3b>(i, j)[2] 这里是获取输入图像的R,G,B 像素灰度值
            I[i][j] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3.0;
            //进行高斯滤波第一步M(x,y) = (Gaussian*(255-I))(x,y)
            inv_I[i][j] = 255;
            *data = inv_I[i][j] - I[i][j];
            data++;
        }
    }
    //这里进行高斯滤波
    GaussianBlur(Mast, Mast, Size(41, 41), BORDER_DEFAULT);
    /**
    @param data:高斯滤波之后的结果
    @param src.at<Vec3b>(i,j)[k]:R,G,B三种颜色灰度值
    
    **/
    Mat dst(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++) {
        uchar* data = Mast.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++) {
                float Exp = pow(2, (128 - data[j]) / 128.0);
                //输出结果
                int value = int(255 * pow(src.at<Vec3b>(i, j)[k] / 255.0, Exp));
                dst.at<Vec3b>(i, j)[k] = value;
            }
        }
    }
    return dst;
}

int main()
{
    Mat img,result;
    img = imread("E://Programing Project//OpenCV//Test//OpenCVtest//test.png",IMREAD_COLOR);

    namedWindow("Result", WINDOW_AUTOSIZE);
    namedWindow("Origin", WINDOW_AUTOSIZE);
    
    result = LCC(img);
    if (result.empty())
    {
        cout << "Error! THE IMAGE IS EMPTY.." << endl;
        return -1;
    }
    else 
    {
        imshow("Origin", img);
        imshow("Result", result);  
    }
    
    waitKey(0);
    return 0;
}





```



