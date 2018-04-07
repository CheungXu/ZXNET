# -*- coding:utf-8 -*-

"""
此模块主要是对一些网络架构计算图的基本操作的封装
"""

import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops
#from tensorflow.contrib.layers.python.layers import batch_norm
from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)




"""
归一化层
x 输入
momentum 衰减动量
epsilon 归一化参数
"""
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def batch_normal(input , scope="scope" , reuse=False):
    return tf.contrib.layers.batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse , updates_collections=None)



"""
卷积通道连接
在特征图维连接调节向量
"""
def conv_cond_concat(x, y):
  
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  #tf.concat(concat_dim, values, name='concat')
  #concat_dim是tensor连接的方向（维度），values是要连接的tensor链表，name是操作名。
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)



"""
二维卷积函数：
input_输入图像
output_dims输出维度（卷积核个数）
k_h、k_w卷积核的高和宽（5，5）
d_h、d_w卷积核在高和宽上的的步长（2，2）
stddev初始化参数的标准差
name层名称

"""
def conv2d(input_, output_dim, 
       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    #初始化权重矩阵w
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    #建立卷积层，padding边缘的填充方式
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    #初始化偏移量向量biases
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    #卷积层输出
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

"""
二维分块通道卷积
"""
def partition_conv(input_, output_dim, block_h = 8, block_w = 8, scope='partition_conv'):
  with tf.variable_scope(scope):
    size = input_.shape.as_list()
    w_step = size[2] // block_w
    h_step = size[1] // block_h
    fms = []
    imgs = []
    #for i in range(w_step):
     # for j in range(h_step):
      #  imgs.append(input_[:,j*block_h:(j+1)*block_h,i*block_w:(i+1)*block_w,:])
    for i in range(w_step):
      fm = []
      for j in range(h_step):
        #img = input_[:,j*block_h:(j+1)*block_h,i*block_w:(i+1)*block_w,:]
        fm.append(conv2d(tf.slice(input_,[0,j*block_h,i*block_w,0],[size[0],block_h,block_w,size[3]]),output_dim,k_h=1,k_w=1,d_h=1,d_w=1,name='pconv_'+str(i)+'_'+str(j)))
      fms.append(fm)

    for i in range(w_step):
      fms[i] = tf.concat(fms[i],2)

    return tf.concat(fms,1)



"""
二维空洞卷积函数
"""
def dilated_conv(input_, output_dim, rate, k_h=3, k_w=3, stddev=0.02, name='dilated_conv'):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.random_normal_initializer(stddev=stddev))
    #空洞卷积
    conv = tf.nn.atrous_conv2d(input_,w,rate,padding='SAME')
    #初始化偏移量向量biases
    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    
    return conv
"""



二维反卷积函数
"""
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

"""
二维最大池化函数
"""
def max_pool(input_, k_h=3, k_w=3, d_h=2, d_w=2, name="maxpool"):
    with tf.variable_scope(name):
      return tf.nn.max_pool(input_,[1,k_h,k_w,1],[1,d_h,d_w,1],padding='SAME')


"""
LReLU激活函数
"""     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


"""
线性分类层函数
input_ 输入
output_size 输出大小
scope 命名空间
stddev 初始化参数的标准差
bias_start 偏置项初始值
with_w 返回值是否包含权重w和b
"""
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, reuse=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear") as scopes:
    if reuse:
      scopes.reuse_variables()
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
