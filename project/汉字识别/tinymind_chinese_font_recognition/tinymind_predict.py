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

width = 32
height = 32

# bn + prelu
def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

# def build_model(out_dims, input_shape=(height, width, 1)):
#     inputs_dim = Input(input_shape)  # inputs_dim的type为tf.Tensor，shape为(None,128,128,1),dtype为float32
#     # model=Sequential()
#     # model.add(Conv2D(32,(3,3),strides=(2, 2), padding='valid')(inputs_dim))

#     x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(
#         inputs_dim)  # “valid”代表只进行有效的卷积，即对边界数据不处理。“VALID”发现余下的窗口不够卷积核大小了所以就把后面的列直接去掉了
#     # conv2D(卷积核个数，卷积核尺寸，步长，填充方式)(输入尺寸)，其中卷积核个数即输出的深度，本次卷积后shape为(None,63,63,32)    (32)

#     x = bn_prelu(x)
#     x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(
#         x)  # “valid”代表只进行有效的卷积，即对边界数据不处理。“VALID”发现余下的窗口不够卷积核大小了所以就把后面的列直接去掉了
#     # conv2D(卷积核个数，卷积核尺寸，步长，填充方式)(输入尺寸)，其中卷积核个数即输出的深度，本次卷积后shape为(None,63,63,32)    (30)
#     x = bn_prelu(x)
#     x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,61,61,32)    (28)
#     x = bn_prelu(x)
#     x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,30,30,32)    (14)

#     x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,28,28,64)    (12)
#     x = bn_prelu(x)
#     x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,26,26,64)    (None,10,10,64)
#     x = bn_prelu(x) 
#     x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,13,13,64)    (None,5,5,64)

#     x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,11,11,128)  (None,3,3,128)
#     x = bn_prelu(x)
#     # x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,5,5,128)         (None,1,1,128)

#     # x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,3,3,128)
#     # x = bn_prelu(x)
#     x = AveragePooling2D(pool_size=(2, 2))(x)  # 本次平均池化后，shape==(None,1,1,128)

#     x_flat = Flatten()(x)  # shape==(None,1*1*128)即(None,128)

#     fc1 = Dense(512)(x_flat)  # Dense（全连接层）,512是该层的输出维度，x_flat是该层的输入维度，则输出shape==(None,512)
#     fc1 = bn_prelu(fc1)
#     dp_1 = Dropout(0.3)(fc1)

#     fc2 = Dense(out_dims)(dp_1)  # 后面会赋值out_dims==100，即100个汉字类别，输出shape==(None,100),None表示样本数
#     fc2 = Activation('softmax')(fc2)

#     model = Model(inputs=inputs_dim, outputs=fc2)
#     return model


#####搭建模型
def build_model(out_dims, input_shape=(32, 32, 1)):
    inputs_dim = Input(input_shape)  # inputs_dim的type为tf.Tensor，shape为(None,128,128,1),dtype为float32
    # model=Sequential()
    # model.add(Conv2D(32,(3,3),strides=(2, 2), padding='valid')(inputs_dim))

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(
        inputs_dim)  # “valid”代表只进行有效的卷积，即对边界数据不处理。“VALID”发现余下的窗口不够卷积核大小了所以就把后面的列直接去掉了
    # conv2D(卷积核个数，卷积核尺寸，步长，填充方式)(输入尺寸)，其中卷积核个数即输出的深度，本次卷积后shape为(None,63,63,32)    (30)

    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,61,61,32)    (28)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,30,30,32)    (14)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,28,28,64)    (12)
    x = bn_prelu(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,26,26,64)    (None,10,10,64)
    x = bn_prelu(x) 
    x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,13,13,64)    (None,5,5,64)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,11,11,128)  (None,3,3,128)
    x = bn_prelu(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)  # 本次池化后，shape==(None,5,5,128)         (None,1,1,128)

    # x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)  # 本次卷积后，shape==(None,3,3,128)
    # x = bn_prelu(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)  # 本次平均池化后，shape==(None,1,1,128)

    x_flat = Flatten()(x)  # shape==(None,1*1*128)即(None,128)

    fc1 = Dense(512)(x_flat)  # Dense（全连接层）,512是该层的输出维度，x_flat是该层的输入维度，则输出shape==(None,512)
    fc1 = bn_prelu(fc1)
    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)  # 后面会赋值out_dims==100，即100个汉字类别，输出shape==(None,100),None表示样本数
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=inputs_dim, outputs=fc2)
    return model

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
    for image in sorted(os.listdir(path)):
        if not image == '.DS_Store':
            image_list.append(path + image)
    return image_list


# ==================================================================================================================================================================-
# read image and resize to gray    加载图片，并对图片进行resize以及归一化处理。该函数只在预测没有标签的数据集时才用到，即本例的'test1'文件夹，
#      而在读取训练集验证集的时候并没有用到该函数。训练的时候读取训练集以及测试集是使用ImageDataGenerator.flow_from_directory()来读取的。
def load_image(image):
    img = Image.open(image)
    img = img.resize((height, width))
    img = np.array(img)  # 难道这里之前的时候img不是array？答：是数组
    img = img / 255
    img = img.reshape((1,) + img.shape + (1,))  # reshape img to size(1, 128, 128, 1)   把二维图片矩阵变为四维的
    return img

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
# save filename , label as csv		# 将图片名称以及其预测类别存为csv文件
def save_csv(test_image_path, predict_label):
    image_list = generator_list_of_imagepath(test_image_path)  # image_list是每张图片的路径组成的列表
    save_arr = np.empty((10000, 2), dtype=np.str)
    save_arr = pd.DataFrame(save_arr, columns=['filename', 'label'])
    print(predict_label)
    predict_label = tran_list2str(predict_label)  # 把图片的预测的汉字标签转换为str形式
    for i in range(len(image_list)):  # 遍历测试集文件夹里面的图片个数次，即测试集有多少张图片，就遍历多少次
        filename = image_list[i].split('/')[-1]  # 对第i张图片的路径字符串进行分割，取路径字符串最后一个'/'分隔符后面的字符串(即图片名称,如1.jpg),赋给filename
        save_arr.values[i, 0] = filename  # 把第张图片文件名图片文件名赋给save_arr的第[i,0]个元素，即图片文件名称置于第0列
        save_arr.values[i, 1] = predict_label[i]  # 把第张图片的预测标签赋给save_arr的第[i,1]个元素，即图片预测的标签置于第1列
    save_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
    print('submit_test.csv have been write, locate is :', os.getcwd())  # os.getcwd() 方法用于返回当前工作目录




# ===============================================================================================
if __name__ == '__main__':
    code_path = '/home/zhz/Downloads/CPS-OCR-Engine-master/ocr/'

    # train_path = code_path + 'dataset_1/train'
    # val_path = code_path + 'dataset_1/test'
    train_path = "/media/zhz/6cff368c-1c54-4cba-aeff-afb6309a2a9a/ctw-baseline/data/new_train_data/"
    val_path = "/media/zhz/6cff368c-1c54-4cba-aeff-afb6309a2a9a/ctw-baseline/data/new_val_data/"

    test_image_path = 'test1/'

    # num_classes = 3755
    num_classes = 3273
    # num_classes = 2602
    # BATCH_SIZE = 64
    WEIGHTS_PATH = 'best_weights_hanzi.hdf5'
    # max_Epochs = 10000

    simple_model = build_model(num_classes)
    print(simple_model.summary())

    print("=====test label=====")
    simple_model.load_weights(WEIGHTS_PATH)
    model = simple_model
    # predict_label = test_image_predict_top_k(model, code_path + test_image_path, train_path, 5)
    predict_label = test_image_predict_top1(model, code_path + test_image_path, train_path)

    print("=====csv save=====")
    save_csv(code_path + test_image_path, predict_label)

    print("====done!=====")

