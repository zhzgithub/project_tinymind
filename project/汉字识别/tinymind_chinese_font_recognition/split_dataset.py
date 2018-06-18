# -*- coding: utf-8 -*-
"""
从数据集中划分train和validation两个文件
train_test_split_ratio=0.1 or 0.2
Tree目录：
    data：
        train：
            folder1
            ......
            folder529
        validation:
            folder1
            ......
            folder529
"""
import os
import random
import PIL.Image as Image		# PIL是Python Imaging Library 图片库


# 检查路径下面是否都是文件
def isfile(path):
    for folder in os.listdir(path):				# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        if not os.path.isdir(path+folder):
            os.remove(path+folder)


# 建立文件夹
def mkdir(path):
    """
    if folder is exists, or make new dir
    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        print(path)
        print('success')
        return True
    else:
        print(path)
        print('folder is exist')
        return False


# 返回文件列表		-----
def eachFile(filepath):
    pathDir = os.listdir(filepath)		# /train路径下的文件和文件夹，本例就是100个类别汉字文件夹
    child_file_name = []
    full_child_file_list = []
    for allDir in pathDir:				#遍历100个类别的汉字文件夹，并不遍历汉字文件夹里面的汉字图片。allDir就是分别取值100个汉字文件夹的名称
        if not allDir == '.DS_Store':	# 若allDir不为'.DS_Store'，则执行下面的合并路径命令。我感觉把'.DS_Store'随便改成别的字符串也能行的样子。
            child = os.path.join(filepath, allDir)			# os.path.join()将多个路径组合后返回。本句话是创建child路径为每个汉字类别的路径

            full_child_file_list.append(child)				# 将每个汉字类别路径添加到full_child_file_list列表
            child_file_name.append(allDir)					# 将每个汉字文件夹的名称添加到child_file_name
    return full_child_file_list, child_file_name			# 返回每个汉字类别路径所组成的full_child_file_list列表，和每个汉字文件夹的名称组成的child_file_name列表


# 转移ratio文件
def move_ratio(data_list, original_str, replace_str):		# 具体作用就是更换路径吗？我感觉这函数名称取得真烂，这函数名称和功能有任何关系吗?
															# 涉及到复制移动图片
    for x in data_list:
        fromImage = Image.open(x)
        x = x.replace(original_str, replace_str)			# x是图片，x.replace(original_str, replace_str)意义何在？---把原来文件夹的路径替换为现在的路径，然后再保存图片
        fromImage.save(x)


if __name__ == '__main__':

    # data_path = 'C:/Users/Jack/Documents/jupyter/font_recognition/chinese_font_recognition/train/'  # 原始数据存放地址
    # data_tra_path = 'C:/Users/Jack/Documents/jupyter/font_recognition/chinese_font_recognition/new_train_val_data/font_tra/'  # new_train_data新取的名
    # data_val_path = 'C:/Users/Jack/Documents/jupyter/font_recognition/chinese_font_recognition/new_train_val_data/font_val/'
	
    data_path = 'E:/BaiduNetdiskDownload/TMD/train/'  # 原始数据存放地址
    data_tra_path = 'E:/BaiduNetdiskDownload/TMD/new_train_val_data/font_tra/'  # new_train_data新取的名
    data_val_path = 'E:/BaiduNetdiskDownload/TMD/new_train_val_data/font_val/'

    full_child_file, child_file = eachFile(data_path)	# 返回每个汉字类别路径所组成的full_child_file列表，和每个汉字文件夹的名称组成的child_file列表

    # 建立相应的文件夹	-------建立了这么多文件夹，后面也没用到啊，那建立了做啥？----用到了，后面用到的路径在前面已经以字符串形式赋值了。
    for i in child_file:			# 遍历每个汉字文件夹的名称组成的child_file列表
        tra_path = data_tra_path + '/' + str(i)		# str(i)是汉字文件夹的名称，感觉中间的这个'/'可以去掉
        mkdir(tra_path)								#	建立训练集文件夹		
        val_path = data_val_path + '/' + str(i)
        mkdir(val_path)								#	建立验证集文件夹

    # 划分train和val
    test_train_split_ratio = 0.9

	
	# 注意一点：不只/train/是路径，/train/a.jpg 也是路径，读取这个路径，也就是读取这个图片文件了
    for i in full_child_file:			# i遍历每个汉字类别路径所组成的full_child_file列表
        pic_dir, pic_name = eachFile(i)		# 返回每个汉字类别路径下的所有汉字图片所组成的pic_dir列表，和每个汉字类别路径下的所有汉字图片的名称组成的pic_name列表
				#由于eachFile()中有一句是child = os.path.join(filepath, allDir)，这一句是合并路径，所以pic_dir应该是汉字图片的路径，如./白/hdjahdksaj.jpg,而并不是读取了图片
        random.shuffle(pic_dir)			# shuffle ：洗牌，搅乱
        train_list = pic_dir[0:int(test_train_split_ratio * len(pic_dir))]	# pic_dir[0 : int(0.9*400)],即pic_dir[0:360]，即把每个汉字类别文件夹下的前360个汉字路径为作为训练集
        val_list = pic_dir[int(test_train_split_ratio * len(pic_dir)):]	#pic_dir[int(0.9*400) : ],即pic_dir[360:],即把每个汉字类别文件夹下的第360到400的汉字路径为作为验证集
													# 以上还仅仅只是对于一个汉字文件夹做的处理，i遍历full_child_file才是对所有的汉字类别做处理。
													# 问题：最后的训练集train_list不是使用的append，对于每次i，train_list，val_list都是重新赋值？
        # train_move, val_move
        print('proprecessing %s' % i)
        # print('train_list:',train_list)
		# # 由于下面代码在for循环中，这意思是对每个类别汉字文件夹的每张图片都要做repalce操作，是更换每张图片名称的操作吗？
        move_ratio(train_list, 'train', 'new_train_val_data/font_tra')	# train_list是某个汉字文件夹的前360张图片路径列表。
																		# 把原来train文件夹下的某类别汉字的前360张图片复制到new_train_val_data/font_tra路径下
        move_ratio(val_list, 'train', 'new_train_val_data/font_val')		# val_list  是某个汉字文件夹的后 40张图片路径列表
   
