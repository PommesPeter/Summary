# 灰度世界算法原理和实现
> 以下大部分引用自:[https://www.yanxishe.com/columnDetail/16881](https://www.yanxishe.com/columnDetail/16881)
> [https://blog.csdn.net/fightingforcv/article/details/47746637](https://blog.csdn.net/fightingforcv/article/details/47746637)

## 前言
这个是一个灰度世界算法的实现，这个算法可以起到白平衡的作用。

## 概念
### 颜色恒常性
颜色恒常性（Color constancy）是指当照射物体表面的颜色光发生变化时，人们对该物体表面颜色的知觉仍然保持不变的知觉特性。转自——[百度百科](https://baike.baidu.com/item/%E9%A2%9C%E8%89%B2%E6%81%92%E5%B8%B8%E6%80%A7/8272543?fr=aladdin)

## 算法原理
人的视觉系统具有**颜色恒常性**，**能从变化的光照环境和成像条件下获取物体表面颜色的不变特性**，但成像设备并不具有这样的调节功能，不同的光照环境会导致采集到的图像颜色与真实颜色存在一定程度的偏差，需要选择合适的**颜色平衡算法**去**消除光照环境对颜色**显示的影响。**（让计算机也能够有颜色恒常性）**
灰度世界算法以灰度世界假设为基础，假设为：对于一幅有着大量色彩变化的图像，RGB 三个分量的平均值趋于同一个灰度值 $\bar Gray$。从物理意思上讲，灰度世界算法假设自然界景物对于光线的**平均反射的均值在整体上是一个定值**，这个定值近似为“灰色”。颜色平衡算法将这一假设强制应用于待处理的图像，可以从图像中**消除环境光的影响，获得原始场景图像。**

## 算法步骤


 - 确定 Gray 有 $2$ 种方法，一种是取固定值，比如最亮灰度值的一般，$8$ 位显示为 $128$。
 - 另一种就是通过计算图像 $R,G,B$ 的三个通道的$\bar R ,\bar G,\bar B$($\bar R ,\bar G,\bar B$代表R,G,B三个分量的平均值)，则$\bar Gray=\frac{\bar R+\bar G+\bar B}{3}$
 - 计算$R,G,B$，$3$ 个通道的增益系数：$k_{r}=\frac{\bar Gray}{R}$,$k_{g}=\frac{\bar Gray}{G}$,$k_{b}=\frac{\bar Gray}{B}$
 - 根据 Von Kries 对角模型，对于图像中的每个像素 $C$，调整其分量 $R,G,B$ 分量：
$$\begin{cases}
C(R')=C(R)*k_{r}\\
C(G')=C(G)*k_{g}\\
C(B')=C(B)*k_{b}
\end{cases}$$

**这种算法简单快速，但是当图像场景颜色并不丰富时，尤其出现大块单色物体时，该算法常会失效。**

**注**：Von Kries提出，可用一个对角矩阵变换描述两种光照条件下同一物体表面颜色间的关系。假定在光照$E(\lambda)_1、E(\lambda)_2$下感知到的物体$R,G,B$值分别为$(p_r^2,p_g^2,p_b^2)、(p_r^1,p_g^1,p_b^1)$

根据Von Kries对角理论有：
$$\begin{bmatrix} 
      p_r^2  \\
      p_g^2  \\
      p_b^2
\end{bmatrix}=D*\begin{bmatrix}   p_r^1  \\
      p_g^1  \\
      p_b^1
\end{bmatrix}=\begin{bmatrix} 
      k_r &0 &0  \\
      0 &k_g& 0  \\
      0 & 0 &k_b
\end{bmatrix}*\begin{bmatrix}   p_r^1  \\
      p_g^1  \\
      p_b^1
\end{bmatrix}
$$

	灰度世界算法（Gray World)是以灰度世界假设为基础的,该假设认为对于一幅有着大量色彩变化的图像, R、 G、 B 三个分量的平均值趋于同一个灰度K。一般有两种方法来确定该灰度。

       （1)直接给定为固定值, 取其各通道最大值的一半,即取为127或128；

       （2)令 K = (Raver+Gaver+Baver)/3,其中Raver,Gaver,Baver分别表示红、 绿、 蓝三个通道的平均值。

         算法的第二步是分别计算各通道的增益：

             Kr=K/Raver;
             Kg=K/Gaver;
             Kb=K/Baver;

         算法第三步为根据Von Kries 对角模型,对于图像中的每个像素R、G、B，计算其结果值：

              Rnew = R * Kr;
              Gnew = G * Kg;
              Bnew = B * Kb;

         对于上式，计算中可能会存在溢出（>255,不会出现小于0的)现象，处理方式有两种。

         a、 直接将像素设置为255，这可能会造成图像整体偏白。

         b、 计算所有Rnew、Gnew、Bnew的最大值，然后利用该最大值将将计算后数据重新线性映射到[0,255]内。实践证明这种方式将会使图像整体偏暗，建议采用第一种方案。


## 算法优缺点
此算法简单快速，但是当图像场景颜色并不丰富时，尤其诗出现大量单色物体时，该算法会失效。

## C++代码
```cpp
Mat GrayWorld(Mat src) {
  vector <Mat> bgr;
  cv::split(src, bgr);
  double B = 0;
  double G = 0;
  double R = 0;
  int row = src.rows;
  int col = src.cols;
  Mat dst(row, col, CV_8UC3);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      B += 1.0 * src.at<Vec3b>(i, j)[0];
      G += 1.0 * src.at<Vec3b>(i, j)[1];
      R += 1.0 * src.at<Vec3b>(i, j)[2];
    }
  }
  B /= (row * col);
  G /= (row * col);
  R /= (row * col);
  printf("%.5f %.5f %.5f\n", B, G, R);
  double GrayValue = (B + G + R) / 3;
  printf("%.5f\n", GrayValue);
  double kr = GrayValue / R;
  double kg = GrayValue / G;
  double kb = GrayValue / B;
  printf("%.5f %.5f %.5f\n", kb, kg, kr);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dst.at<Vec3b>(i, j)[0] = (int)(kb * src.at<Vec3b>(i, j)[0]) > 255 ? 255 : (int)(kb * src.at<Vec3b>(i, j)[0]);
      dst.at<Vec3b>(i, j)[1] = (int)(kg * src.at<Vec3b>(i, j)[1]) > 255 ? 255 : (int)(kg * src.at<Vec3b>(i, j)[1]);
      dst.at<Vec3b>(i, j)[2] = (int)(kr * src.at<Vec3b>(i, j)[2]) > 255 ? 255 : (int)(kr * src.at<Vec3b>(i, j)[2]);
    }
  }
  return dst;
}

```
#### spilt函数

> 该函数解释引自：[https://blog.csdn.net/alickr/article/details/51503133](https://blog.csdn.net/alickr/article/details/51503133)
> 作者：alickr 

split函数的功能是通道分离原型

```cpp
void split(const Mat& src,Mat *mvBegin)

void split(InputArray m, OutputArrayOfArrays mv);
```
用法很显然，第一个参数为要进行分离的图像矩阵，第二个参数可以是Mat数组的首地址，或者一个vector<Mat>对象
```cpp
std::vector<Mat> channels;
Mat aChannels[3];
//src为要分离的Mat对象
split(src, aChannels);              //利用数组分离
split(src, channels);             //利用vector对象分离
 
imshow("B",channels[0]);
imshow("G",channels[1]);
imshow("R",channels[2]);
```
**$注意:opencv中，RGB三个通道是反过来的——BGR$**

可能有人会问为什么分离出的通道都是黑白灰，而不是红绿蓝。原因是分离后为单通道，相当于分离通道的同时把其他两个通道填充了相同的数值。比如红色通道，分离出红色通道的同时，绿色和蓝色被填充为和红色相同的数值，这样一来就只有黑白灰了。那么红色体现在哪呢？可以进行观察，会发现原图中颜色越接近红色的地方在红色通道越接近白色。

在纯红的地方在红色通道会出现纯白。

R值为255 -》RGB(255，255，255)，为纯白
## 代码解释

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat GrayWorld(Mat src) {
    //利用vector对象分离输入图像的R,G,B三个通道
    vector <Mat> bgr;
    split(src, bgr);
    double B = 0;
    double G = 0;
    double R = 0;
    int row = src.rows;
    int col = src.cols;
    //创建与输入图像同等大小的Mat对象
    Mat dst(row, col, CV_8UC3);
    //获取输入图像每一个像素的RGB值
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            B += 1.0 * src.at<Vec3b>(i, j)[0];
            G += 1.0 * src.at<Vec3b>(i, j)[1];
            R += 1.0 * src.at<Vec3b>(i, j)[2];
        }
    }
    //求平均
    B /= (row * col);
    G /= (row * col);
    R /= (row * col);
    printf("%.5f %.5f %.5f\n", B, G, R);
    //计算图像RGB三个通道值取平均值
    double GrayValue = (B + G + R) / 3;
    printf("%.5f\n", GrayValue);
    //计算三个通道的增益系数
    double kr = GrayValue / R;
    double kg = GrayValue / G;
    double kb = GrayValue / B;
    printf("%.5f %.5f %.5f\n", kb, kg, kr);
    //根据Von Kries对角模型。对于图像中每一个像素C调整都调整其RGB分量
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            dst.at<Vec3b>(i, j)[0] = (int)(kb * src.at<Vec3b>(i, j)[0]) > 255 ? 255 : (int)(kb * src.at<Vec3b>(i, j)[0]);
            dst.at<Vec3b>(i, j)[1] = (int)(kg * src.at<Vec3b>(i, j)[1]) > 255 ? 255 : (int)(kg * src.at<Vec3b>(i, j)[1]);
            dst.at<Vec3b>(i, j)[2] = (int)(kr * src.at<Vec3b>(i, j)[2]) > 255 ? 255 : (int)(kr * src.at<Vec3b>(i, j)[2]);
        }
    }
    return dst;
}

int main()
{
    Mat img, result;
    img = imread("E://Programing Project//OpenCV//Test//OpenCVtest//test.png", IMREAD_COLOR);

    namedWindow("Result", WINDOW_AUTOSIZE);
    namedWindow("Origin", WINDOW_AUTOSIZE);

    result = GrayWorld(img);
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

可以看到灰度世界算法有了白平衡的效果，并且该算法的执行速度也是非常的快。