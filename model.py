# -*- coding:utf-8 -*-  

from __future__ import division #精确除
import os
import time
import math
from glob import glob #查找路径
import tensorflow as tf
import numpy as np
from six.moves import xrange 

from ops import *
from utils import *


"""
（math.ceil 向上取整）
返回以same方式填充的卷积后图像大小
size：原图像大小
stride：步长
返回值：卷积后图像大小

"""
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


"""
深度卷积对抗生成神经网络类

"""
class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default', mask_type = 'default', sampleset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,learning_rate_init = 0.0001):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for Z. [128]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess  #TF会话
    self.crop = crop  #输入输出大小是否一致

    self.batch_size = batch_size #批大小
    self.sample_num = sample_num #样本个数

    self.input_height = input_height #输入图像高
    self.input_width = input_width #输入图像宽
    self.output_height = output_height #输出图像高
    self.output_width = output_width #输出图像宽

    self.gf_dim = gf_dim #G的第一个卷积层通道数
    self.df_dim = df_dim #D的第一个卷积层通道数

    self.gfc_dim = gfc_dim #G的第一个全连接层维数
    self.dfc_dim = dfc_dim #D的第一个全连接层维数

    #数据集名称
    self.dataset_name = dataset_name
    #遮挡模式
    self.mask_type = mask_type
    #样本集名称
    self.sampleset_name = sampleset_name
    #图像文件格式
    self.input_fname_pattern = input_fname_pattern
    #检查点标记
    self.checkpoint_dir = checkpoint_dir

    #使用glob得到数据路径，imread载入数据。
    #c_dim：图像channel数。
    self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
    self.c_dim = imread(self.data[0]).shape[-1]

    self.mask_data = [os.path.join("./data", self.dataset_name+'_'+self.mask_type, path.split('/')[3]) for path in self.data]

    self.samples_names = glob(os.path.join("./data", self.sampleset_name, '*.png'))
    self.samples_mask = [os.path.join("./data", self.sampleset_name+'_'+self.mask_type, path.split('/')[3]) for path in self.samples_names]
    
    #初始化学习率
    self.learn_rate_init = learning_rate_init

    #得到图像是否为灰度图
    self.grayscale = (self.c_dim == 1)
    #建立模型
    self.build_model()

  def build_model(self):
    #crop标记为True，图像大小为输出大小；
    #否则为输入大小。
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    #添加输入层占位符和样本输入占位符，占位符为4维向量：
    #第1维：batch中图像位置
    #第2维：图像高度
    #第3维：图像宽度
    #第4维：图像channels
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    self.mask_inputs = tf.placeholder(
      tf.float32, [self.batch_size,self.input_height, self.input_width, self.c_dim], name='mask_images')

    inputs = self.inputs
    mask_inputs = self.mask_inputs

    #按照相应的方法建立网络。
    self.mask_z = self.encoder(mask_inputs)
    self.inputs_z = self.encoder(inputs, reuse=True)

    self.x_tilde= self.generator(self.mask_z)


    self.local_x_tilde = self.x_tilde[:,32:96,18:82,:]
    self.local_inputs = self.inputs[:,32:96,18:82,:]
    self.local_width = 64
    self.local_height = 64

    self.De_pro_tilde = self.discriminator(self.x_tilde)
    self.local_De_pro_tilde = self.local_discriminator(self.local_x_tilde)
  
    self.D_pro_logits = self.discriminator(inputs, reuse=True)
    self.local_D_pro_logits = self.local_discriminator(self.local_inputs, reuse=True)

    self.D_logits = linear(tf.concat([self.D_pro_logits,self.local_D_pro_logits],1), 1, 'd_log_res_lin')
    self.D_tilde = linear(tf.concat([self.De_pro_tilde,self.local_De_pro_tilde], 1), 1, 'd_log_res_lin',reuse=True)

    #获取交叉熵损失函数
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    #D的损失函数
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D_logits)))
    self.d_loss_tilde = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_tilde, tf.zeros_like(self.D_tilde)))
    #D的loss为真假loss之和                      
    self.D_loss = self.d_loss_real + self.d_loss_tilde

    #特征损失
    self.z_loss = self.NLLNormal2(self.mask_z, self.inputs_z) / ((self.input_height / 4) * (self.input_width / 4) * 256) 
    self.LL_loss = self.NLLNormal2(self.x_tilde, self.inputs) / ((self.input_height) * (self.input_width) * 3)
    self.local_LL_loss = self.NLLNormal2(self.local_x_tilde, self.local_inputs) / ((self.local_height) * (self.local_width) * 3)

    #编码器损失
    self.encode_loss = - self.z_loss - self.LL_loss - self.local_LL_loss 

    #G的损失函数
    self.G_loss_tilde = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_tilde, tf.ones_like(self.D_tilde)))
    self.G_loss = self.G_loss_tilde - self.LL_loss

 
    #用scalar_summary记录日志
    #记录图像。
    self.input_sum = image_summary('Inputs',self.inputs)
    self.G_sum = image_summary("G", self.x_tilde)
    self.local_input_sum = image_summary('Local_Input',self.local_inputs)
    self.local_G_sum = image_summary("Local_G", self.local_x_tilde)

    #记录损失函数
    self.z_loss_sum = scalar_summary("z_loss", -self.z_loss)
    self.ll_loss_sum = scalar_summary("ll_loss", -self.LL_loss)
    self.local_ll_loss_sum = scalar_summary("local_ll_loss", -self.local_LL_loss)

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_tilde_sum = scalar_summary("d_loss_tilde", self.d_loss_tilde)

    self.e_loss_sum = scalar_summary("e_loss", self.encode_loss)
    self.g_loss_sum = scalar_summary("g_loss", self.G_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.D_loss)


    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.e_vars = [var for var in t_vars if 'e_' in var.name]
    #添加存储器
    self.saver = tf.train.Saver(max_to_keep = 0)

  def train(self, config):
    #添加学习率衰减
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=1000,
                                                   decay_rate=0.98)
    #D优化器
    #opti_D = tf.train.GradientDescentOptimizer(new_learning_rate).minimize(self.D_loss)
    trainer_D = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
    gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
    opti_D = trainer_D.apply_gradients(gradients_D)

    #G优化器
    #opti_G = tf.train.AdamOptimizer(new_learning_rate, beta1=config.beta1).minimize(self.G_loss, var_list=self.g_vars)
    trainer_G = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
    gradients_G = trainer_G.compute_gradients(self.G_loss, var_list=self.g_vars)
    opti_G = trainer_G.apply_gradients(gradients_G)

    #E优化器
    #opti_E = tf.train.AdamOptimizer(new_learning_rate, beta1=config.beta1).minimize(self.encode_loss, var_list=self.e_vars)
    trainer_E = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
    gradients_E = trainer_E.compute_gradients(self.encode_loss, var_list=self.e_vars)
    opti_E = trainer_E.apply_gradients(gradients_E)

    #初始化变量
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    #执行总结并记录入log中
    self.g_sum = merge_summary(
        [self.input_sum, self.G_sum, self.d_loss_tilde_sum, self.local_input_sum, self.local_G_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.d_loss_real_sum, self.d_loss_sum])
    self.e_sum = merge_summary(
        [self.ll_loss_sum, self.z_loss_sum, self.e_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)
    
    #开始训练
    counter = 0
    start_time = time.time()
    #尝试载入检查点
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    #载入sample数据
    sample_files = self.mask_data[0:self.sample_num]
    sample_real_files = self.data[0:self.sample_num]
    samples = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for sample_file in sample_files]
    sample_real = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for sample_file in sample_real_files]
    #合并通道
    #mask_img = get_image(os.path.join("./data",self.dataset_name+'_'+self.mask_type,'mask.'+self.input_fname_pattern.split('.')[1]),
     #                    input_height=self.input_height,input_width=self.input_width,resize_height=self.input_height,resize_width=self.input_width,grayscale=True)
    #new_mask = mask_img[:,:,np.newaxis]
    sample = samples #[np.concatenate([im,new_mask],axis=2) for im in samples]

    #转换数据类型
    if (self.grayscale):
      sample_real_save = np.array(sample_real).astype(np.float32)[:, :, :, None]
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      sample_inputs_save = np.array(samples).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
      sample_real_save = np.array(sample_real).astype(np.float32)
      sample_inputs_save = np.array(samples).astype(np.float32)

    #保存测试图像
    save_images(sample_inputs_save, image_manifold_size(sample_inputs_save.shape[0]),
          './{}/train_input.png'.format(config.sample_dir))
    save_images(sample_real_save, image_manifold_size(sample_real_save.shape[0]),
          './{}/train_real.png'.format(config.sample_dir))

    #开始迭代
    #将图置为只读
    self.sess.graph.finalize() 
    for epoch in xrange(counter,config.epoch):
      # 计算batch数，// 为整数除 
      batch_idxs = min(len(self.data)-self.sample_num, config.train_size) // config.batch_size
      
      for idx in xrange(0, batch_idxs):
        #获取当前batch图像数据
        batch_files = self.data[self.sample_num + idx * config.batch_size:self.sample_num + (idx+1) * config.batch_size]
        mask_batch_files = self.mask_data[self.sample_num + idx * config.batch_size:self.sample_num + (idx+1) * config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        mask_batchs = [
            get_image(mask_batch_files,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for mask_batch_files in mask_batch_files]
        mask_batch = mask_batchs #[np.concatenate([im,new_mask],axis=2) for im in mask_batchs]
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          mask_images = np.array(mask_batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)
          mask_images = np.array(mask_batch).astype(np.float32)

        # 更新生成器
        _, summary_str = self.sess.run([opti_G,self.g_sum],feed_dict={self.inputs: batch_images,self.mask_inputs: mask_images})
        self.writer.add_summary(summary_str, epoch)

        # 更新编码器
        _, summary_str = self.sess.run([opti_E,self.e_sum],feed_dict={self.inputs: batch_images,self.mask_inputs: mask_images})
        self.writer.add_summary(summary_str, epoch)

        # 更新判别器
        _, summary_str = self.sess.run([opti_D,self.d_sum],feed_dict={self.inputs: batch_images,self.mask_inputs: mask_images})
        self.writer.add_summary(summary_str, epoch)
        
        # 更新学习率
        if new_learn_rate > 0.00005:
          self.sess.run(add_global) 
        new_learn_rate = self.sess.run(new_learning_rate)
        
        # 输出损失
        D_loss, fake_loss, encode_loss, LL_loss, z_loss, new_learn_rate = self.sess.run([self.D_loss, self.G_loss, self.encode_loss,self.LL_loss, self.z_loss, new_learning_rate], feed_dict={self.inputs:batch_images,self.mask_inputs: mask_images})
        print("Epochs %d/%d Batch %d/%d: D: loss = %.7f G: loss=%.7f E: loss=%.7f LL loss=%.7f Z loss=%.7f, LR=%.7f" % (epoch, config.epoch, idx, batch_idxs,D_loss, fake_loss, encode_loss, LL_loss, z_loss, new_learn_rate))
      
      # 保存图像
      if np.mod(epoch, 1) == 0:
        sample_outputs= self.sess.run(
              self.x_tilde,
              feed_dict={self.mask_inputs: sample_inputs})

        save_images(sample_outputs, image_manifold_size(sample_outputs.shape[0]),
              './{}/train_output{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

        print("[Sample] : %s" % ('./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx)))

      # 保存模型
      if np.mod(epoch, 1) == 0:
        self.save(config.checkpoint_dir, epoch)
        print("Save checkpoint in : %s, counter: %d" % (config.checkpoint_dir,epoch))


  def discriminator(self, image, reuse=False):
    #判别器D，reuse表示是否复用参数。
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      #建立网络
      #df_dim为第一层卷积层维数（卷积核个数），默认64.
      #函数conv2d/lrelu/linear/batch_normal见ops.py模块

      """
      网络结构：（图像大小以输入128×128×3为例）
      卷积核大小默认5*5，步长2，bs为batch_size。
      第一层卷积：h0，卷积核64个，输入图像大小bs*128*128*3。
      第二层卷积：h1，卷积核128个，输入图像大小bs*64*64*64。
      第三层卷积：h2，卷积核256个，输入图像大小bs*32*32*128。
      第四层卷积：h3，卷积核512个，输入图像大小bs*16*16*256。
      第五层卷积：h4，卷积核512个，输入图像大小bs*8*8*512，输出图像大小bs*4*4*512。
      第六层全连接：h5，输入bs*8192, 输出bs*1024。
      """

      h0 = lrelu(batch_normal(conv2d(image, self.df_dim, k_h=5, k_w=5, d_h=2, d_w=2, name='gl_d_h0_conv'),scope='gl_d_bn0', reuse=reuse)) #64*64*64
      h1 = lrelu(batch_normal(conv2d(h0, self.df_dim*2, k_h=5, k_w=5, d_h=2, d_w=2, name='gl_d_h1_conv'),scope='gl_d_bn1', reuse=reuse)) #32*32*128
      h2 = lrelu(batch_normal(conv2d(h1, self.df_dim*4, k_h=5, k_w=5, d_h=2, d_w=2, name='gl_d_h2_conv'),scope='gl_d_bn2', reuse=reuse)) #16*16*256
      h3 = lrelu(batch_normal(conv2d(h2, self.df_dim*8, k_h=5, k_w=5, d_h=2, d_w=2, name='gl_d_h3_conv'),scope='gl_d_bn3', reuse=reuse)) #8*8*512
      h4 = lrelu(batch_normal(conv2d(h3, self.df_dim*8, k_h=5, k_w=5, d_h=2, d_w=2, name='gl_d_h4_conv'),scope='gl_d_bn4', reuse=reuse)) #4*4*512
      
      h5 = lrelu(batch_normal(linear(tf.reshape(h4, [self.batch_size, -1]), 1024, 'l_d_h5_lin'), scope='l_d_bn5', reuse=reuse)) #1024
 
      #返回结果
      return h5

  def local_discriminator(self, image, reuse=False):
    #局部判别器D，reuse表示是否复用参数。
    with tf.variable_scope("local_discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      #建立相应的网络
      #df_dim为第一层卷积层纬度（卷积核个数），默认64.
      #函数conv2d/lrelu/linear见ops.py模块

      """
      网络结构：（图像大小以输入64×64×3为例）
      卷积核大小默认5*5，步长2，bs为batch_size。
      第一层卷积：h0，卷积核64个，输入图像大小bs*64*64*3。
      第二层卷积：h1，卷积核256个，输入图像大小bs*32*32*64。
      第三层卷积：h2，卷积核512个，输入图像大小bs*16*16*512。
      第四层卷积：h3，卷积核512个，输入图像大小bs*8*8*512，, 输出图像大小bs*4*4*512。
      第五层全连接：h4，输入bs*8192, 输出bs*1024。
      """
      h0 = lrelu(batch_normal(conv2d(image, self.df_dim, name='l_d_h0_conv',k_h=5, k_w=5, d_h=2, d_w=2),scope='l_d_bn0',reuse=reuse)) #32*32*64
      h1 = lrelu(batch_normal(conv2d(h0, self.df_dim*4, k_h=5, k_w=5, d_h=2, d_w=2,name='l_d_h1_conv'), scope='l_d_bn1', reuse=reuse)) #16*16*256
      h2 = lrelu(batch_normal(conv2d(h1, self.df_dim*8, k_h=5, k_w=5, d_h=2, d_w=2, name='l_d_h2_conv'), scope='l_d_bn2', reuse=reuse)) #8*8*512
      h3 = lrelu(batch_normal(conv2d(h2, self.df_dim*8, k_h=5, k_w=5, d_h=2, d_w=2, name='l_d_h3_conv'), scope='l_d_bn3', reuse=reuse)) #4*4*512

      h4 = lrelu(batch_normal(linear(tf.reshape(h3, [self.batch_size, -1]), 1024, 'l_d_h4_lin'), scope='l_d_bn4', reuse=reuse))

      #返回结果
      return h4

  def encoder(self,image,reuse=False):
    with tf.variable_scope("encoder") as scope:
      if reuse:
        scope.reuse_variables()
      """
      网络结构：（图像大小以输入128×128×3为例）
      第一层卷积：h0，卷积核64个，输入图像大小bs*128*128*3。

      第二层卷积：h1，卷积核128个，输入图像大小bs*128*128*64。
      第三层卷积：h2，卷积核128个，输入图像大小bs*64*64*128。

      第四层卷积：h3，卷积核256个，输入图像大小bs*64*64*128。
      第五层卷积：h4，卷积核256个，输入图像大小bs*32*32*256。

      第六层卷积：h5，卷积核256个，输入图像大小bs*32*32*256。
      第七层扩展卷积：h6，卷积核256个，输入图像大小bs*32*32*256，扩展率2。
      第八层扩展卷积：h7，卷积核256个，输入图像大小bs*32*32*256，扩展率4。
      第九层扩展卷积：h8，卷积核256个，输入图像大小bs*32*32*256，扩展率8。
      第十层扩展卷积：h9，卷积核256个，输入图像大小bs*32*32*256，扩展率16。
      """

      h0 = lrelu(batch_normal(conv2d(image, self.df_dim, name='e_h0_conv',k_h=5, k_w=5), scope='e_bn0',reuse=reuse)) #128*128*64
      
      h1 = lrelu(batch_normal(conv2d(h0, self.df_dim*2, name='e_h1_conv', k_h=3, k_w=3, d_h = 2,d_w = 2), scope='e_bn1',reuse=reuse)) #64*64*128
      h2 = lrelu(batch_normal(conv2d(h1, self.df_dim*2, name='e_h2_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='e_bn2',reuse=reuse)) #64*64*128

      h3 = lrelu(batch_normal(conv2d(h2, self.df_dim*4, name='e_h3_conv', k_h=3, k_w=3, d_h = 2,d_w = 2), scope='e_bn3',reuse=reuse)) #32*32*256     
      h4 = lrelu(batch_normal(conv2d(h3, self.df_dim*4, name='e_h4_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='e_bn4',reuse=reuse)) #32*32*256 
      
      h5 = lrelu(batch_normal(conv2d(h4, self.df_dim*4, name='e_h5_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='e_bn5',reuse=reuse)) #32*32*256 
      h6 = lrelu(batch_normal(dilated_conv(h5, self.df_dim*4, rate=2, name='e_h6_dilconv'), scope='e_bn6',reuse=reuse)) #32*32*256 
      h7 = lrelu(batch_normal(dilated_conv(h6, self.df_dim*4, rate=4, name='e_h7_dilconv'), scope='e_bn7',reuse=reuse)) #32*32*256 
      h8 = lrelu(batch_normal(dilated_conv(h7, self.df_dim*4, rate=8, name='e_h8_dilconv'), scope='e_bn8',reuse=reuse)) #32*32*256 
      h9 = lrelu(batch_normal(dilated_conv(h8, self.df_dim*4, rate=16, name='e_h9_dilconv'), scope='e_bn9',reuse=reuse)) #32*32*256 
      
      return h9
 
  def generator(self, input_, reuse=False):
    #生成器G，参数z为属于噪声，y为标签。
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()

      """
      s_h/s_w输出图像高宽（128，128）
      s_h2/s_w2前一层高宽（64，64）
      s_h4/s_h4前两层高宽（32，32）
      s_h8/s_h8前三层高宽（16，16）
      s_h16/s_h16前四层高宽（8，8)
      """
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


      """
      第一层卷积： h0, 卷积核256个，输入大小bs×32*32*256。

      第二层反卷积：h1, 卷积核256个，输入大小bs*32*32*256。
      第三层卷积：h2, 卷积核128个，输入大小bs*64*64*256

      第四层反卷积：h3, 卷积核128个，输入大小bs*64*64*128。
      第五层卷积： h4, 卷积核64个，输入入大小bs*128*128*128。

      第六层卷积： h5, 卷积核32个，输入大小bs*128*128*64。
      第七层卷积： h6, 卷积核3个，输入大小bs*128*128*32, 输出大小bs*128*128*3。
      """
      
      h0 = lrelu(batch_normal(conv2d(input_, self.df_dim*4, name='e_h0_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='e_bn0',reuse=reuse)) #32*32*256

      h1 = lrelu(batch_normal(conv2d(h0, self.df_dim*4, name='e_h1_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='e_bn1',reuse=reuse)) #64*64*256
      h2 = lrelu(batch_normal(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*2], k_h=4, k_w=4, name='g_h2_deconv'),scope='g_bn2',reuse=reuse)) #64*64*128

      h3 = lrelu(batch_normal(conv2d(h2, self.df_dim*2, name='g_h3_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='g_bn3',reuse=reuse)) #128*128*128
      h4 = lrelu(batch_normal(deconv2d(h3, [self.batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h4_deconv'),scope='g_bn4',reuse=reuse)) #128*128*64

      h5 = lrelu(batch_normal(conv2d(h4, self.df_dim/2, name='g_h5_conv', k_h=3, k_w=3, d_h = 1,d_w = 1), scope='g_bn5',reuse=reuse)) #128*128*32
      h6 = conv2d(h5, 3, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h6_comv') #128*128*3

      #返回tanh激活后的结果。
      return tf.nn.tanh(h6)

  def NLLNormal(self, pred, target):
      c = -0.5 * tf.log(2 * np.pi)
      multiplier = 1.0 / (2.0 * 1)
      tmp = tf.square(pred - target)
      tmp *= -multiplier
      tmp += c

      return tmp

  def NLLNormal2(self, pred, target):
      return -tf.reduce_sum(tf.square(pred - target))
  @property #将方法变为属性
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
