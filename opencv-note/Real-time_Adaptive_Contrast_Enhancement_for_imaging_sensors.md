# 《Real-time adaptive contrast enhancement for imaging sensors》论文解读及实现

> 以下内容参考：[https://mp.weixin.qq.com/s/aPi7haF7eTDabbYi4c75cA](https://mp.weixin.qq.com/s/aPi7haF7eTDabbYi4c75cA)

## 前言 
这个 ACE 算法是用来做**图像对比度增强的算法**。图像对比度增强的算法在很多场合都有用处，特别是在医学图像中，这是因为在众多疾病的诊断中，医学图像的视觉检查时很有必要的。医学图像由于本身及成像条件的限制，图像的**对比度**很低。因此，在这个方面已经开展了很多的研究。这种增强算法一般都遵循一定的**视觉原则**。众所周知，人眼对**高频信号**（边缘处等）比较敏感。虽然细节信息往往是高频信号，但是他们时常嵌入在大量的低频背景信号中，从而使得其视觉可见性降低。因此适当的**提高高频部分能够提高视觉效果**并有利于诊断。

## 算法原理

> 个人理解：

 一张图片，总是由**低频部分**和**高频部分**构成的，**低频部分可以由图像的低通滤波来得到**，而**高频部分可以由原图减去低频部分来得到**。而本算法的目标是**增强代表细节的高频部分**，即是对**高频部分乘上一个系数**，然后重组得到增强的图像。所以本算法的核心就是**高频部分增益系数(又叫 CG)的计算**，第一种方法是将这个系数设为一个常数，第二种方法是将**增益表示为与方差相关的量**。假设图像中的某个点表示为$x(i,j)$，那么以$(i,j)$为中心，窗口大小为$(2n+1) × (2n+1)$,其局部均值和局部方差为：
 $$m_x(i,j)=\frac{1}{(2n+1)^2}\sum_{k=i-n}^{i+n}\sum_{l=j-n}^{j+n}x(k,l)$$
 $$\sigma^2(i,j)=\frac{1}{(2n+1)^2}\sum_{k=i-n}^{i+n}\sum_{l=j-n}^{j+n}[x(k,l)-m_x(i,j)]^2$$
 上面的式子中$\sigma(i,j)$就是所谓的局部标准差(LSD)，定义$f(i,j)$表示$x(i,j)$对应的增强后的像素值.
 
  ACE 算法可以表示为：$f(i, j)=m_x(i, j)+G(i,j)[x(i,j)-m_x(i,j)]$
  
其中系数$G(i,j)$就是上面说的 $CG$。一般情况下 $CG$ 总是大于 $1$ 的，这样高频部分就可以得到增强。$CG$ 的取值有 $2$ 种，一种是直接取一个常数 $C(C>1)$，这样上面的式子可以写成：

$f(i, j)=m_x(i, j)+C[x(i, j)-m_x(i, j)]$, 其中$C$ 是一个大于 $1$ 的数。

这种情况下，图像中所有的高频部分都被同等放大，可能有些高频部分会出现过增强的现象的。
而第二种方法是对每个位置使用不同的增益，Lee 等人提出了下面的解决方案：

$$f(i, j)=m_x(i, j)+\frac{D}{\sigma_x(i,j)}[x(i,j)-m_x(i,j)]$$

其中 $D$ 是一个常数，这样$CG$ 系数是空间自适应的，并且和局部均方差成反比，在图像的边缘或者其他变化剧烈的地方，局部均方差比较大，因此 $CG$ 的值就比较小，这样就不会产生振铃效应。然而，在平滑的区域，局部均方差就会很小，这样 $CG$ 的值比较大，从而引起了噪音的放大，所以需要对 $CG$ 的最大值做一定的限制才能获得更好的效果。

我们使用第二种方法，因为在图像的高频区域，局部方差较大，此时增益值就较小，这样就不会出现过亮的情况。但是在图像平滑的区域，局部均方差很小，此时增益值较大，从而可能会方法噪声信号，所以需要对增益最大值做一定的限制。$D$ 这个常数一些文章认为取图像的全局均值，而我这里参考 ImageShop 大牛的文章使用了全局均方差。下面给出一些代码实现和效果测试。

## C++代码

```cpp
//自适应对比度增强算法，C表示对高频的直接增益系数,n表示滤波半径，maxCG表示对CG做最大值限制
Mat ACE(Mat src, int C = 3, int n = 3, float MaxCG = 7.5){
    int row = src.rows;
    int col = src.cols;
    Mat meanLocal; //图像局部均值
    Mat varLocal; //图像局部方差
    Mat meanGlobal; //全局均值
    Mat varGlobal; //全局标准差
    blur(src.clone(), meanLocal, Size(n, n));
    Mat highFreq = src - meanLocal;
    varLocal = highFreq.mul(highFreq);
    varLocal.convertTo(varLocal, CV_32F);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            varLocal.at<float>(i, j) = (float)sqrt(varLocal.at<float>(i, j));
        }
    }
    meanStdDev(src, meanGlobal, varGlobal);
    Mat gainArr = meanGlobal / varLocal; //增益系数矩阵
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(gainArr.at<float>(i, j) > MaxCG){
                gainArr.at<float>(i, j) = MaxCG;
            }
        }
    }
    printf("%d %d\n", row, col);
    gainArr.convertTo(gainArr, CV_8U);
    gainArr = gainArr.mul(highFreq);
    Mat dst1 = meanLocal + gainArr;
    //Mat dst2 = meanLocal + C * highFreq;
    return dst1;
}

int main(){
    Mat src = imread("../test.png");
    vector <Mat> now;
    split(src, now);
    int C = 150;
    int n = 5;
    float MaxCG = 3;
    Mat dst1 = ACE(now[0], C, n, MaxCG);
    Mat dst2 = ACE(now[1], C, n, MaxCG);
    Mat dst3 = ACE(now[2], C, n, MaxCG);
    now.clear();
    Mat dst;
    now.push_back(dst1);
    now.push_back(dst2);
    now.push_back(dst3);
    cv::merge(now, dst);
    imshow("origin", src);
    imshow("result", dst);
    imwrite("../result.jpg", dst);
    waitKey(0);
    return 0;
}

```

## 效果:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200224185806111.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70)

## 代码解释

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
//高频=原图-低频 低频通过低通滤波得到
//低通滤波(Low-pass filter) 是一种过滤方式，规则为低频信号能正常通过，而超过设定临界值的高频信号则被阻隔、减弱。但是阻隔、减弱的幅度则会依据不同的频率以及不同的滤波程序（目的）而改变。
//自适应对比度增强算法，C表示对高频的直接增益系数,n表示滤波半径，maxCG表示对CG做最大值限制
Mat ACE(Mat src, int C = 3, int n = 3, float MaxCG = 7.5) 
{
    int row = src.rows;
    int col = src.cols;
    Mat meanLocal; //图像局部均值
    Mat varLocal; //图像局部方差
    Mat meanGlobal; //全局均值
    Mat varGlobal; //全局标准差
    //blur()函数可以用标准化的盒式过滤器来平滑图像。
    //https://blog.csdn.net/duwangthefirst/article/details/79971322
    //copyTo 也是深拷贝，但是否申请新的内存空间，取决于dst矩阵头中的大小信息是否与src一至，若一致则只深拷贝并不申请新的空间，否则先申请空间后再进行拷贝．
    //clone 是完全的深拷贝，在内存中申请新的空间
    //https://blog.csdn.net/u013806541/article/details/70154719?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    blur(src.clone(), meanLocal, Size(n, n));
    //求高频部分
    Mat highFreq = src - meanLocal;
    //dot为矩阵点乘，即对应元素相乘再相加(要求A、B行列式相同)
    //mul是对应位的乘积,这里是highFreq每一个元素都平方
    //https://blog.csdn.net/dcrmg/article/details/52404580
    varLocal = highFreq.mul(highFreq);
    //convertTo()矩阵数据类型转换:https://blog.csdn.net/iracer/article/details/49204147
    varLocal.convertTo(varLocal, CV_32F);
    for (int i = 0; i < row; i++) 
    {
        for (int j = 0; j < col; j++) 
        {
            //求标准差
            varLocal.at<float>(i, j) = (float)sqrt(varLocal.at<float>(i, j));
        }
    }
    //计算矩阵所有元素的均值和标准差。https://blog.csdn.net/hk121/article/details/83109994
    meanStdDev(src, meanGlobal, varGlobal);
    Mat gainArr = meanGlobal / varLocal; //增益系数矩阵
    for (int i = 0; i < row; i++) 
    {
        for (int j = 0; j < col; j++) 
        {
            if (gainArr.at<float>(i, j) > MaxCG) 
            {
                gainArr.at<float>(i, j) = MaxCG;
            }
        }
    }
    printf("%d %d\n", row, col);
    gainArr.convertTo(gainArr, CV_8U);
    gainArr = gainArr.mul(highFreq);
    Mat dst1 = meanLocal + gainArr;
    //Mat dst2 = meanLocal + C * highFreq;
    //dst为目标图像
    return dst1;
}

int main() {
    Mat src = imread("test4.png");
    vector <Mat> now;
    split(src, now);
    int C = 150;
    int n = 5;
    float MaxCG = 3;
    Mat dst1 = ACE(now[0], C, n, MaxCG);
    Mat dst2 = ACE(now[1], C, n, MaxCG);
    Mat dst3 = ACE(now[2], C, n, MaxCG);
    now.clear();
    Mat dst;
    now.push_back(dst1);
    now.push_back(dst2);
    now.push_back(dst3);
    //该函数用来合并通道
    merge(now, dst);
    imshow("origin", src);
    imshow("result", dst);
    imwrite("../result.jpg", dst);
    waitKey(0);
    return 0;
}

```


论文地址：[https://www.researchgate.net/publication/253622155_Real-Time_Adaptive_Contrast_Enhancement_For_Imaging_Sensors](https://www.researchgate.net/publication/253622155_Real-Time_Adaptive_Contrast_Enhancement_For_Imaging_Sensors)


