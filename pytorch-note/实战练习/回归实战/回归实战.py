import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.autograd import *


# 计算loss
def compute_error_for_line_give_points(b, w, points):  # 计算损失
    """
    b:当前的 b值
    w:当前的 w值
    points:list 点
    """
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


# 计算梯度
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_decent_runner(points, starting_b, starting_w,
                           learningRate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learningRate)

    return [b, w]

# 运行代码
def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.00001
    init_b = 0
    init_w = 0
    num_iterations = 1000
    print('Starting gradient descent at b = %f, w = %f, error =%f' % (
        init_b, init_w, compute_error_for_line_give_points(init_b, init_w, points)))
    print('Running...')
    [b, w] = gradient_decent_runner(points, init_b, init_w, learning_rate, num_iterations)
    print('After %d iterations b = %f, w = %f, error = %f' % (
        num_iterations, b, w, compute_error_for_line_give_points(b, w, points)))


if __name__ == '__main__':
    run()
