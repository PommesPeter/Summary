# 自动白平衡之完美反射算法

> 以下内容引自：[https://www.chainnews.com/articles/414282117914.htm](https://www.chainnews.com/articles/414282117914.htm)

## 算法原理
此算法的原理非常简单，完美反射理论假设图像中**最亮的点就是白点**，并以此**白点为参考对图像进行自动白平衡**，**最亮点定义为的最大值**

## 算法步骤
- 计算每个像素之和$R+G+B$并保存。
- 按照的值的大小计算出其前 `10%` 或其他 `Ratio` 的白色参考点的阈值 $T$。
- 遍历图像中的每个点，计算$R+G+B$大于 $T$ 的所有点的$R、G、B$分量的累积和的平均值。
- 将每个像素量化到$[0,255]$

## C++代码

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat PerfectReflectionAlgorithm(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	int HistRGB[767] = { 0 };
	int MaxVal = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[0]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[1]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[2]);
			int sum = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			HistRGB[sum]++;
		}
	}
	int Threshold = 0;
	int sum = 0;
	for (int i = 766; i >= 0; i--) {
		sum += HistRGB[i];
		if (sum > row* col * 0.1) {
			Threshold = i;
			break;
		}
	}
	int AvgB = 0;
	int AvgG = 0;
	int AvgR = 0;
	int cnt = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int sumP = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			if (sumP > Threshold) {
				AvgB += src.at<Vec3b>(i, j)[0];
				AvgG += src.at<Vec3b>(i, j)[1];
				AvgR += src.at<Vec3b>(i, j)[2];
				cnt++;
			}
		}
	}
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
			int Green = src.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
			int Red = src.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
			if (Red > 255) {
				Red = 255;
			}
			else if (Red < 0) {
				Red = 0;
			}
			if (Green > 255) {
				Green = 255;
			}
			else if (Green < 0) {
				Green = 0;
			}
			if (Blue > 255) {
				Blue = 255;
			}
			else if (Blue < 0) {
				Blue = 0;
			}
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}

int main()
{
	Mat img, result;
	img = imread("E://Programing Project//OpenCV//Test//OpenCVtest//grayworldtest.jpg", IMREAD_COLOR);

	namedWindow("Result", WINDOW_AUTOSIZE);

	namedWindow("Origin", WINDOW_AUTOSIZE);

	result = PerfectReflectionAlgorithm(img);
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
## 代码解释

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat PerfectReflectionAlgorithm(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	int HistRGB[767] = { 0 };
	int MaxVal = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			//找到图像中最亮的点，且定义为最大值
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[0]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[1]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[2]);
			//计算每一个像素之和
			//@param sum:每一个像素之和
			int sum = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			HistRGB[sum]++;
		}
	}
	int Threshold = 0;
	int sum = 0;
	for (int i = 766; i >= 0; i--) {
		sum += HistRGB[i];
		//按照的值的大小计算出其前 10% 或其他 Ratio 的白色参考点的阈值 T。
		if (sum > row* col * 0.1) {
			Threshold = i;
			break;
		}
	}
	
	int AvgB = 0;
	int AvgG = 0;
	int AvgR = 0;
	int cnt = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int sumP = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			if (sumP > Threshold) {
				AvgB += src.at<Vec3b>(i, j)[0];
				AvgG += src.at<Vec3b>(i, j)[1];
				AvgR += src.at<Vec3b>(i, j)[2];
				cnt++;
			}
		}
	}
	//计算R+G+BR+G+B大于 T 的所有点的R、G、B分量的累积和的平均值。
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	//将每一个像素量化到[0,255]
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
			int Green = src.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
			int Red = src.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
			if (Red > 255) {
				Red = 255;
			}
			else if (Red < 0) {
				Red = 0;
			}
			if (Green > 255) {
				Green = 255;
			}
			else if (Green < 0) {
				Green = 0;
			}
			if (Blue > 255) {
				Blue = 255;
			}
			else if (Blue < 0) {
				Blue = 0;
			}
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;

			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}

int main()
{
	Mat img, result;
	img = imread("E://Programing Project//OpenCV//Test//OpenCVtest//grayworldtest.jpg", IMREAD_COLOR);

	namedWindow("Result", WINDOW_AUTOSIZE);

	namedWindow("Origin", WINDOW_AUTOSIZE);

	result = PerfectReflectionAlgorithm(img);
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

可以看到自动白平衡算法之完美反射算法算法有了白平衡的效果，并且该算法的执行速度也是非常的快。