# 利用多尺度融合提升图像细节

> 以下内容大部分引用自：[https://zhuanlan.zhihu.com/p/105109282](https://zhuanlan.zhihu.com/p/105109282)

## 前言
今天为大家介绍一个利用多尺度来提升图像细节的算法。这个算法来自于论文《DARK IMAGE ENHANCEMENT BASED ON PAIRWISE TARGET CONTRAST AND MULTI-SCALE DETAIL BOOSTING》

## 算法原理
核心就是，论文使用了Retinex方法类似的思路，使用了多个尺度的高斯核对原图滤波，然后再和原图做减法，获得不同程度的细节信息，然后通过一定的组合方式把这些细节信息融合到原图中，从而得到加强原图信息的能力。公式十分简单，注意到第一个系数有点特殊，实现的话，直接看下图的几个公式即可。

「从深度学习中特征金字塔网络的思想来看，这个算法实际上就是将不同尺度上的特征图进行了融合，不过这个方式是直接针对原图进行，比较粗暴，但有个好处就是这个算法用于预处理阶段是易于优化的，关于如何优化后面讲SSE指令集优化的时候再来讨论，今天先提供原始的实现啦。」
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200222174527762.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTcwOTMzMA==,size_16,color_FFFFFF,t_70)

## C++代码

```cpp
void separateGaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma){
    CV_Assert(src.channels()==1 || src.channels() == 3); //只处理单通道或者三通道图像
    //生成一维的
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for(int i = 0; i < ksize; i++){
        double g = exp(-(i-origin) * (i-origin) / (2 * sigma * sigma));
        sum += g;
        matrix[i] = g;
    }
    for(int i = 0; i < ksize; i++) matrix[i] /= sum;
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BORDER_CONSTANT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    //水平方向
    for(int i = border; i < rows; i++){
        for(int j = border; j < cols; j++){
            double sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<uchar>(i, j+k);
                }else if(channels == 3){
                    Vec3b rgb = dst.at<Vec3b>(i, j+k);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    //竖直方向
    for(int i = border; i < rows; i++){
        for(int j = border; j < cols; j++){
            double sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<uchar>(i+k, j);
                }else if(channels == 3){
                    Vec3b rgb = dst.at<Vec3b>(i+k, j);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete [] matrix;
}

Mat MultiScaleDetailBoosting(Mat src, int Radius){
    int rows = src.rows;
    int cols = src.cols;
    Mat B1, B2, B3;
    separateGaussianFilter(src, B1, Radius, 1.0);
    separateGaussianFilter(src, B2, Radius*2-1, 2.0);
    separateGaussianFilter(src, B3, Radius*4-1, 4.0);
    float w1 = 0.5, w2 = 0.5, w3 = 0.25;
    Mat dst(rows, cols, CV_8UC3);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < 3; k++){
                int D1 = src.at<Vec3b>(i, j)[k] - B1.at<Vec3b>(i, j)[k];
                int D2 = B1.at<Vec3b>(i, j)[k] - B2.at<Vec3b>(i, j)[k];
                int D3 = B2.at<Vec3b>(i, j)[k] - B3.at<Vec3b>(i, j)[k];
                int sign = D1 > 0 ? 1 : -1;
                dst.at<Vec3b>(i, j)[k] = saturate_cast<uchar>((1 - w1*sign) * D1 - w2 * D2 + w3 * D3 + src.at<Vec3b>(i, j)[k]);
            }
        }
    }
    return dst;
}
```
