from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Activation, Embedding
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, PReLU, Lambda
from keras.models import Model, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import keras.backend as K
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# bn + prelu
def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


# ==================================================================================================================================================================-
# learning rate of epoch	# 这里的epoch没用到啊，后面的程序直接调用的lr = LearningRateScheduler(lrschedule)，根本没有给参数epoch赋值！
def lrschedule(epoch):
    if epoch <= 40:
        return 0.1
    elif epoch <= 80:
        return 0.01
    else:
        return 0.001


# label for directory in disk 每类汉字的标签
# def label_of_directory(directory):		# 这里的directory最后被赋值为train_path = code_path + 'new_train_val_data/font_tra/'
# """
# sorted for label indices
# return a dict for {'classes', 'range(len(classes))'}
# """
# classes = []
# for subdir in sorted(os.listdir(directory)):	# 对汉字文件夹路径排序
# if os.path.isdir(os.path.join(directory, subdir)):		# 如果汉字文件夹是路径，就把汉字文件夹路径添加进classes列表中    os.path.join() 将多个路径组合后返回
# classes.append(subdir)

# num_classes = len(classes)  # 统计训练集文件夹下的汉字文件夹有多少个，即汉字类别有多少。明显是100个
# class_indices = dict(zip(classes, range(len(classes)))) # 生成 汉字文件夹路径作为关键字，汉字类别数目作为键值 的字典，即{类别:索引}
# return class_indices	# 函数返回值为{类别:索引}的字典形式


# # 读取数据
# # 直接读取数据到内存？还是先占据个位置训练的时候再分批次读取进内存？
# # 训练的时候分批次读取进内存，该函数需要在训练或预测模块函数内使用，不能直接在训练之外使用，不然会一直占据大量内存
# def read_data(path):
# image_list=[]
# label_list=[]
# for img_dir in sorted(os.listdir(path)):	# 遍历图片文件夹，而非图片
# label_list.append(img_dir)
# for image in os.listdir(path+img_dir):		# 遍历每个图片文件夹的图片
# # print(image)
# img=load_image(path+img_dir+'/'+image)
# # print(img.shape)
# image_list.append(img)
# image_arr=np.array(image_list)
# label_list=sorted(label_list*len(os.listdir(path+img_dir)))
# # label_arr=np.array(label_list).reshape(len(label_list),1)		# 这里只是每个文件夹的标签，并非文件夹下的图片的标签

# # print(image_arr.shape)			 #shape=(?,128,128,1)
# return image_arr,label_list		# 函数返回值为所有图片组成的数组，但是并没有标签

# # 标签变为onehot形式的标签
# def trans2onehot_label(labels,columns):	# 这里的label是由所有样本的标签组成的标签向量,标签向量为[1,3,8,12,..]的形式

# new_label=np.zeros((len(labels),100),dtype=np.uint8)  # 创建一个样本数行，100列的零矩阵

# # 例如：某标签为12的话，那么第12个位置就设置为1
# count=0
# for value in labels:
# for j in range(100):
# if value==j:
# new_label[count][value]=1
# break
# count+=1
# return new_label


# 搭建模型
def build_model(out_dims, input_shape=(128, 128, 1)):
    inputs_dim = Input(input_shape)  # inputs_dim的type为tf.Tensor，shape为(None,128,128,1),dtype为float32
    # model=Sequential()
    # model.add(Conv2D(32,(3,3),strides=(2, 2), padding='valid')(inputs_dim))

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(
        inputs_dim)  # “valid”代表只进行有效的卷积，即对边界数据不处理。“VALID”发现余下的窗口不够卷积核大小了所以就把后面的列直接去掉了
    # conv2D(卷积核个数，卷积核尺寸，步长，填充方式)(输入尺寸)，其中卷积核个数即输出的深度，本次卷积后shape为(None,63,63,32)

    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,61,61,32)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,30,30,32)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,28,28,64)
    x = bn_prelu(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,26,26,64)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,13,13,64)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,11,11,128)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,5,5,128)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,3,3,128)
    x = bn_prelu(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)  # 本次平均池化后，shape==(None,1,1,128)

    x_flat = Flatten()(x)  # shape==(None,1*1*128)即(None,128)

    fc1 = Dense(512)(x_flat)  # Dense（全连接层）,512是该层的输出维度，x_flat是该层的输入维度，则输出shape==(None,512)
    fc1 = bn_prelu(fc1)
    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)  # 后面会赋值out_dims==100，即100个汉字类别，输出shape==(None,100),None表示样本数
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=inputs_dim, outputs=fc2)
    return model


# 训练模型
# def train_model(istrain):
# if not istrain:
# if os.path.exists(weights_path):
# model.load_weights(weights_path)
# else:
# print('weights文件不存在，不能加载权重文件，请重新修改train_model函数的istrain参数')
# else:
# print('开始训练')
# model=construct_model(100,input_shape=(128,128,1))
# sgd = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
# model.compile(optimizer='sgd',
# loss='categorical_crossentropy',
# metrics=['accuracy'])
# # model.fit(data, labels)  # starts training 这种训练方式是把所有的训练数据一起输入进去，没有分batch
# for epochs in range(100):
# train_loss=model.train_on_batch(x_train,y_train)
# # model.predict(x_val,y_val)
# val_loss,val_acc=model.evaluate(x_val,y_val)
# print(('epoch=%d-----train_loss=%f---val_loss=%f---val_acc=%f',(epochs,train_loss,val_loss,val_acc)))
# model.save(weights_path)
# return model

def model_train(model):
    lr = LearningRateScheduler(lrschedule)
    mdcheck = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', save_best_only=True)  # 在每个epoch后保存模型到WEIGHTS_PATH
    td = TensorBoard(log_dir=code_path + 'new_train_val_data/tensorboard_log/')

    sgd = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    print("model compile!!")
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    '''
        model.compile(optimizer, loss, metrics=None, sample_weight_mode=None)  
        编译用来配置模型的学习过程，其参数有
        optimizer：字符串（预定义优化器名）或优化器对象，参考优化器 
        loss：字符串（预定义损失函数名）或目标函数，参考损失函数
        metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
        sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。在下面fit函数的解释中有相关的参考内容。
        kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano作为后端，kwargs的值将会传递给 K.function

        如实例：
        model.compile( loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'] )  
    '''
    print("model training!!")
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=32000 // BATCH_SIZE,
                                  # steps_per_epoch=128 // BATCH_SIZE,
                                  epochs=max_Epochs,
                                  validation_data=val_generator,
                                  # validation_steps=128 // BATCH_SIZE,
                                  validation_steps=8000 // BATCH_SIZE,
                                  callbacks=[lr, mdcheck, td])
    return history
'''				http://keras-cn.readthedocs.io/en/latest/models/sequential/#fit_generator
            fit_generator
			fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, 
							validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
			利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
            函数的参数是：

            generator：生成器函数，生成器的输出应该为：
            一个形如（inputs，targets）的tuple
            一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
            steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
            epochs：整数，数据迭代的轮数
            verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
            validation_data：具有以下三种形式之一

            生成验证集的生成器
            一个形如（inputs,targets）的tuple
            一个形如（inputs,targets，sample_weights）的tuple
            validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数
            class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
            sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
            workers：最大进程数
            max_q_size：生成器队列的最大容量
            pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
            initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
            函数返回一个History对象
        '''

# 以下定义的这么多函数都是为了处理最后要预测的数据集，对于训练时并不需要使用到这些函数。本例即是"test1"文件夹下的数据
# 以下是进行预测时候的步骤
'''  进行预测时候的步骤：
def test_image_predict_top1():	# 从训练好的模型中加载权重文件进行预测,这里只预测一个最优类别,函数返回值为"test1"图片文件夹下所有图片的汉字类别组成的列表
def test_image_predict_top_k():	# 从训练好的模型中加载权重文件进行预测，这里预测top_5个最优类别，返回汉字形式的所有图片的预测标签组成的列表

	1. def generator_list_of_imagepath()		# 把path路径下的所有图片的路径加入到image_list列表中，该列表的元素是图片的路径，如./a/1.jpg

	2. def label_of_directory()		# 把汉字标签映射为数字标签{类别:索引}，如{ 白:0, 高:1, ...}的形式

	3. def load_image()				# 加载图片，并对图片进行resize以及归一化处理

	4.	def get_label_predict_top1()	# 获取预测图片的标签：输入的参数image是图片，返回值是该图片的标签，是一个数字
		def get_label_predict_top_k()	# 获取预测图片的前top_k个最大可能的标签，返回值为top_k个数字组成的列表。

	5. def get_key_from_value()		# 把索引值等于字典的键值的对应汉字标签关键字找出来


# test_image_predict_top_k()预测出的标签是列表形式，即每个标签中间隔开了，因此需要把列表形式的标签转换为汉字标签不间隔的字符串形式，然后存在csv文件中
def save_csv():
	def tran_list2str()				# 把list转换为str形式，由于本来一张图片预测出来了5个标签，这5个标签构成每张图片的对应标签列表里面的的5个元素，
			# 这五个元素是分隔开的，最后提交的文件格式是无间隔的几个汉字。因此需要把这五个元素以字符串形式连接起来
'''

# ==================================================================================================================================================================-
# label for directory in disk  该函数功能是返回 把汉字文件夹名称作为关键字，汉字类别索引作为键值 的字典，即把汉字标签映射为数字标签{类别:索引}，如{ 白:0, 高:1, ...}的形式
def label_of_directory(directory):  # 这里的directory最后被赋值为train_path = code_path + 'new_train_val_data/font_tra/'
    """
    sorted for label indices
    return a dict for {'classes', 'range(len(classes))'}
    """
    classes = []
    for subdir in sorted(os.listdir(directory)):  # 对汉字文件夹目录排序。这里的目录不是绝对路径或相对路径，而是列出汉字文件夹的名称，如"白"、"高"，而不是./train/"白"
        if os.path.isdir(
                os.path.join(directory, subdir)):  # 如果汉字文件夹是目录					 os.path.join() 将多个路径组合后返回
            classes.append(subdir)				   # 就把汉字文件夹目录添加进classes列表中，这里的汉字文件夹目录是如 "白"的形式。

    num_classes = len(classes)  # 统计训练集文件夹下的汉字文件夹有多少个，即汉字类别有多少。明显是100个
    class_indices = dict(zip(classes, range(len(classes))))  # 生成 汉字文件夹名称作为关键字，汉字类别索引作为键值 的字典，即{类别:索引}
    return class_indices  # 函数返回值为{类别:索引}的字典形式，如{ 白:0, 高:1, ...}的形式


# zip()的用法：zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# >>>a = [1,2,3]
# >>> b = [4,5,6]
# >>> c = [4,5,6,7,8]
# >>> zipped = zip(a,b)     # 打包为元组的列表
# [(1, 4), (2, 5), (3, 6)]

# >>> dict(a='a', b='b', t='t')     # 传入关键字
# {'a': 'a', 'b': 'b', 't': 't'}
# >>> dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # 映射函数方式来构造字典
# {'three': 3, 'two': 2, 'one': 1} 
# >>> dict([('one', 1), ('two', 2), ('three', 3)])    # 可迭代对象方式来构造字典
# {'three': 3, 'two': 2, 'one': 1}


# ==================================================================================================================================================================-
# get key from value in dict   该函数功能是把索引值等于字典的键值的对应汉字标签关键字找出来，因为后面做预测时候需要的是预测出汉字类别，而不是他对应的数字标签
def get_key_from_value(dict, index):
    for keys, values in dict.items():
        if values == index:
            return keys


# ==================================================================================================================================================================-
# 该函数功能是把path路径下的所有图片的路径加入到image_list列表中，该列表的元素是图片的路径，如./a/1.jpg
# 这是把"test1"文件夹下的所有图片路径添加进列表
def generator_list_of_imagepath(path):
    image_list = []
    for image in os.listdir(path):
        if not image == '.DS_Store':
            image_list.append(path + image)
    return image_list


# ==================================================================================================================================================================-
# read image and resize to gray    加载图片，并对图片进行resize以及归一化处理。该函数只在预测没有标签的数据集时才用到，即本例的'test1'文件夹，
#      而在读取训练集验证集的时候并没有用到该函数。训练的时候读取训练集以及测试集是使用ImageDataGenerator.flow_from_directory()来读取的。
def load_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))
    img = np.array(img)  # 难道这里之前的时候img不是array？答：是数组
    img = img / 255
    img = img.reshape((1,) + img.shape + (1,))  # reshape img to size(1, 128, 128, 1)   把二维图片矩阵变为四维的
    return img


# ==================================================================================================================================================================-
# 获取预测图片的标签：输入的参数image是图片，返回值是该图片的标签，是一个数字，因为中间把onehot形式的标签变为了一个数字，所以返回值是一个数字，将来会把它从字典映射回汉字标签
def get_label_predict_top1(image, model):
    """
    image = load_image(image), input image is a ndarray
    retturn best of label
    """
    predict_proprely = model.predict(image)  # 这里会预测出该张图片的数字标签，是onehot形式，而不是汉字标签
    predict_label = np.argmax(predict_proprely, axis=1)  # 找出每行中的最大值位置的索引值，即是汉字标签映射的键值。即把onehot形式的输出值转化为单个数字的输出值
    return predict_label  # 这里的标签是一个数字


# ==================================================================================================================================================================-
# 获取预测图片的前top_k个最大可能的标签，返回值为top_k个数字组成的列表。
# 输入的参数image是图片，返回值是该图片的标签，是数字形式的标签列表，因为中间把onehot形式的标签变为了一个数字，所以返回值是五个预测数字标签的列表，将来会把它从字典映射回汉字标签
def get_label_predict_top_k(image, model, top_k):
    """
    image = load_image(image), input image is a ndarray
    return top-5 of label
    """
    # array 2 list		把数组转化为列表
    predict_proprely = model.predict(image)  # 这里会预测出该张图片的数字标签，是onehot形式，而不是汉字标签
    predict_list = list(predict_proprely[0])  # 因为输出维度为(None,100)的二维矩阵，所以a[0]取得是一行，而a[0][2]是第0行第2列的一个元素
    # 对于一张图片进行预测，它的输出应该是一个(1,100)矩阵，而不是一个(100,)序列向量。但是不添加[0]也行啊，那干嘛还添加个[0]呢？
    # 如果是(1,100)的矩阵，那么list(predict_proprely[0])就是[array([0,0,...,1,...,0])]的形式了，咋办？说到底还是一个序列向量。
    min_label = min(predict_list)  # 找出标签中的最小值，在下面的程序中会用到
    label_k = []
    for i in range(top_k):  # 循环top_k次，每次找出predict_list中的最大值的索引值，然后把该索引值对应的值(即最大值)从列表中remove
        label = np.argmax(predict_list)  # 把predict_list中的最大值的索引值赋给label
        predict_list.remove(predict_list[label])  # 把最大值的索引值对应的值(即最大值)从predict_list列表中remove
        predict_list.insert(label, min_label)  # 在移除最大值的位置处，把列表中的最小值插入到该位置
        label_k.append(label)  # 把最大值的索引值添加到label_k列表中
    return label_k  # 经过for循环，最终label_k列表中会有top_k个值，即softmax输出的值从大到小排列的top_k个值的索引值。这些索引值将来就会映射为汉字标签


# ==================================================================================================================================================================-
#  从训练好的模型中加载权重文件进行预测,这里只预测一个最优类别,函数返回值为"test1"图片文件夹下所有图片的汉字类别组成的列表
def test_image_predict_top1(model, test_image_path, directory):
    model.load_weights(WEIGHTS_PATH)  # 加载训练好的权重文件到模型
    image_list = generator_list_of_imagepath(test_image_path)  # 把测试集(这里指的是"./test1/"文件夹)要预测的图片的路径加入到image_list列表中

    predict_label = []
    class_indecs = label_of_directory(directory)  # {类别：索引}的字典class_indecs,即把汉字类别映射为数字索引
    for image in image_list:  # 遍历image_list列表中的每个元素，即每张图片的路径，如./a/1.jpg
        img = load_image(image)  # 加载一张图片，并做resize和归一化处理
        label_index = get_label_predict_top1(img, model)  # 获取预测图片的标签，这里的标签是一个数字，将来会把它从字典映射回汉字标签
        label = get_key_from_value(class_indecs,
                                   label_index)  # 当label_index和class_indecs字典的value值相等时,返回class_indecs的keys。即找出预测出的数字标签对应的汉字
        predict_label.append(label)  # 将一张图片预测出的汉字形式的类别（标签）加入到predict_label列表中
    return predict_label  # 函数返回值为整个图片文件夹下的图片的汉字类别


# ==================================================================================================================================================================-
#  从训练好的模型中加载权重文件进行预测，这里预测top_5个最优类别，返回汉字形式的所有图片的预测标签组成的列表
def test_image_predict_top_k(modle, test_image_path, directory, top_k):
    model.load_weights(WEIGHTS_PATH)  # 由于训练完成后会生成权重文件，因此这里进行预测的时候就得加载训练后的权重文件
    image_list = generator_list_of_imagepath(test_image_path)  # 把测试集(这里指的是"./test1/"文件夹)要预测的图片的路径加入到image_list列表中

    predict_label = []
    class_indecs = label_of_directory(directory)  # {类别：索引}的字典class_indecs,即把汉字类别映射为数字索引
    # 这个外循环是对文件夹下的所有图片组成的列表进行遍历
    for image in image_list:  # 遍历image_list列表中的每个元素，即每张图片的路径，如./a/1.jpg
        img = load_image(image)  # 加载一张图片，并做resize和归一化处理
        # return a list of label max->min
        label_index = get_label_predict_top_k(img, model,
                                              5)  # 获取预测图片的前top_k个最大概率的标签，这里的label_index是top_k个数字组成的列表，将来会把它从字典映射回汉字标签
        label_key_list = []  #
        # 下面这个内循环是对一张图片的五个预测标签进行遍历，把索引值标签转化为汉字标签
        for label in label_index:  # 因为这里标签有5个候选值，因此需要把这5个候选值对应的汉字找出来，所以需要遍历每个标签索引值
            label_key = get_key_from_value(class_indecs,
                                           label)  # 当label和class_indecs字典的value值相等时，返回class_indecs的keys。即找出预测出的数字标签对应的汉字
            label_key_list.append(str(label_key))  # 把索引值对应的关键字即汉字标签添加进label_key_list列表，该列表最终会有top_k个元素。
        # 注意这里需要把汉字转换为str。但是top1的时候他却没有转换为str? 后面有专门的tran_list2str()会把列表转换为str

        predict_label.append(label_key_list)  # 把内循环得到的一张图片的预测标签(top_k个元素组成的列表)添加进predict_label列表

    return predict_label  # 函数返回值为所有图片的预测标签组成的predict_label列表，该列表的每个元素也是列表，即一张图片的top_k个预测标签(汉字形式)


# ==================================================================================================================================================================-
# translate list to str in label  把list转换为str形式
def tran_list2str(predict_list_label):
    new_label = []
    for row in range(len(predict_list_label)):  # 遍历所有图片的预测出的标签组成的列表
        str = ""  # 这里不是表示成对双引号，而是指无空格，即双引号之间没有空格，即汉字之间没有分隔符
        for label in predict_list_label[row]:  # 遍历一张图片的预测标签列表的top_k个预测标签
            str += label  # 遍历完一次内循环后，str为 你是谁啊你
        new_label.append(str)  # 然后把 你是谁啊你   添加进new_label列表
    return new_label  # 返回值为[你是谁啊你,你是谁啊你,你是谁啊你,...]的形式


# ==================================================================================================================================================================-
# save filename , label as csv		# 将图片名称以及其预测类别存为csv文件
def save_csv(test_image_path, predict_label):
    image_list = generator_list_of_imagepath(test_image_path)  # image_list是每张图片的路径组成的列表
    save_arr = np.empty((10000, 2), dtype=np.str)
    save_arr = pd.DataFrame(save_arr, columns=['filename', 'label'])
    predict_label = tran_list2str(predict_label)  # 把图片的预测的汉字标签转换为str形式
    for i in range(len(image_list)):  # 遍历测试集文件夹里面的图片个数次，即测试集有多少张图片，就遍历多少次
        filename = image_list[i].split('/')[-1]  # 对第i张图片的路径字符串进行分割，取路径字符串最后一个'/'分隔符后面的字符串(即图片名称,如1.jpg),赋给filename
        save_arr.values[i, 0] = filename  # 把第张图片文件名图片文件名赋给save_arr的第[i,0]个元素，即图片文件名称置于第0列
        save_arr.values[i, 1] = predict_label[i]  # 把第张图片的预测标签赋给save_arr的第[i,1]个元素，即图片预测的标签置于第1列
    save_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
    print('submit_test.csv have been write, locate is :', os.getcwd())  # os.getcwd() 方法用于返回当前工作目录


# # 预测一个正确的：top1
# def predict_top1(test_image_path):
# model=train_model(path,istrain)
# model.load_weights(weights_path)
# predict_image=[]
# for image in test_image_path:  # 遍历测试集里面的所有图片
# predict=model.predict(image)
# predict_image.append(predict)
# return [image,predict_image]


# # 预测5个正确的：top5	
# def predict_top5(path):
# model=train_model(path,istrain)
# model.load_weights(weights_path)
# model.predict(x_test)


# # 输出结果文件
# def save_csv(path):
# data_arr=np.array(predict_top1(test_image_path),dtype=np.str)
# data_arr = np.empty((10000, 2), dtype=np.str)
# data_arr=pd.DataFrame(data_arr,columns=['filename','label'])
# data_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
# print('submit_test.csv have been write, locate is :', os.getcwd())		# os.getcwd() 方法用于返回当前工作目录


if __name__ == '__main__':
	code_path = 'E:/BaiduNetdiskDownload/TMD/'
	train_path = code_path + 'new_train_val_data/font_tra/'
	val_path = code_path + 'new_train_val_data/font_val/'
	test_image_path = 'test1/'

	num_classes = 100
	BATCH_SIZE = 128
	WEIGHTS_PATH = 'best_weights_hanzi.hdf5'
	max_Epochs = 100

	# ImageDataGenerator就是图片数据生成器，下面函数是构建训练集生成器
	train_datagen = ImageDataGenerator(  # https://blog.csdn.net/weiwei9363/article/details/78635674
		rescale=1. / 255,  # rescale值在执行其他处理前乘到整个图像上，这个值定为0~1之间的数
		# horizontal_flip=True		#进行随机水平翻转
		width_shift_range=0.15,  # 宽度偏移,随机沿着水平方向，以图像的宽小部分百分比为变化范围进行平移;
		height_shift_range=0.15  # 高度偏移,随机沿着垂直方向，以图像的高小部分百分比为变化范围进行平移;
	)

	val_datagen = ImageDataGenerator(  # 对验证集只需要对像素值缩放1/255就行
		rescale=1. / 255
	)

# 利用keras中image.ImageDataGenerator.flow_from_directory()实现从文件夹中提取图片和进行简单归一化处理，以及数据扩充。但是标签没有读进去啊？
# 答：自动读取了标签，每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性class_indices可获得文件夹名与类的序号的对应字典。
# 所以使用flow_from_directory()给定训练集或者验证集路径之后，他就自动读取数据及其标签，不用管它怎么分的类。
# 对训练集图像进行数据扩充	利用上面构建的训练集生成器来对图片进行归一化以及数据扩充。
# 但是到底有没有扩充呢？图片数量并没有增加啊？难道是只增加在内存中，并没有存到本地文件夹里面？
# 答：他没有实现扩充后的存储操作，应该是只做了数据扩充，却没有执行扩充让他读取进数据集。
	train_generator = train_datagen.flow_from_directory(  # https://blog.csdn.net/u012193416/article/details/79368855

		train_path,  # 训练集的路径
		target_size=(128, 128),  # 在这里即对训练集的图片进行缩放，缩放为（128,128）
		batch_size=BATCH_SIZE,  # 在这里对训练设置batch_size
		color_mode='grayscale',
		class_mode='categorical',		# class_mode参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签
		# save_to_dir=code_path+ 'gen/' ,
		# save_prefix='gen'
	)
''' http://keras-cn.readthedocs.io/en/latest/preprocessing/image/#imagedatagenerator
1.flow_from_directory()
flow_from_directory(directory): 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据

directory: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用.详情请查看此脚本
target_size: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
color_mode: 颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
classes: 可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性class_indices可获得文件夹名与类的序号的对应字典。
class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
batch_size: batch数据的大小,默认32
shuffle: 是否打乱数据,默认为True
seed: 可选参数,打乱数据和进行变换时的随机数种子
save_to_dir: None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
flollow_links: 是否访问子文件夹中的软链接

2.flow()
flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')：接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据

x：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
y：标签
batch_size：整数，默认32
shuffle：布尔值，是否随机打乱数据，默认为True
save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
seed: 整数,随机数种子
'''
	
	# 以上用的是ImageDataGenerator.flow_from_directory()函数,该函数对图片进行简单归一化处理。本程序并没有使用数据扩充,因为总样本还是40000个。数据生成后并没有保存下来读取。
	# 还有一种是使用ImageDataGenerator.flow()函数，该函数将会返回一个生成器，这个生成器用来扩充数据，每次都会产生batch_size个样本。 https://blog.csdn.net/weiwei9363/article/details/78635674

# 对验证集进行数据扩充    但是标签没有读进去啊？
#答：自动读取了标签，因为读取每个文件夹下的图片就已经知道了他们分别是哪一个类别。所以使用flow_from_directory()给定训练集或者验证集路径之后，他就自动读取数据及其标签，不用管它怎么分的类。
	val_generator = val_datagen.flow_from_directory(
		val_path,
		target_size=(128, 128),
		batch_size=BATCH_SIZE,
		color_mode='grayscale',
		class_mode='categorical'			# class_mode参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签
	)

	simple_model = build_model(num_classes)
	print(simple_model.summary())

	print("=====start train image of epoch=====")

	model_history = model_train(simple_model)

	print("=====show acc and loss of train and val====")
	# draw_loss_acc(model_history)

	print("=====test label=====")
	simple_model.load_weights(WEIGHTS_PATH)
	model = simple_model
	predict_label = test_image_predict_top_k(model, code_path + test_image_path, train_path, 5)

	print("=====csv save=====")
	save_csv(code_path + test_image_path, predict_label)

	print("====done!=====")
