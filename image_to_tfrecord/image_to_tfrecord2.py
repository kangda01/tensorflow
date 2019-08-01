import glob
import os.path
import numpy as np
import tensorflow as tf
from PIL import Image


#@profile
def image_to_tfrecord(image_path, tfrecord_path):
    #os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    sub_dirs = [x[0] for x in os.walk(image_path)][1:]
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    #读取所有的子目录
    for lable, sub_dir in enumerate(sub_dirs):
        extensions = ['jpg', 'jpeg']
        file_list = []
        #os.path.basename函数返回文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(image_path, dir_name, '*.'+extension)
            #glob.glob(pathname)返回所有匹配的文件路径列表。它只有一个参数pathname，
            # 定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        #处理图片数据
        for i, file_name in enumerate(file_list):
            #读取并解析图片， 将图片转换成299*299，以便inception-v3处理
            image_raw_data = Image.open(file_name)
            image_raw_data =image_raw_data.resize((299,299),Image.ANTIALIAS)
            image_raw = image_raw_data.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'lable': _int64_frature(lable),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
            # print("写入成功：",file_name)
    writer.close()


# 生成整数型的属性
def _int64_frature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
    image_path = r"G:\tensorflow_google\chapter_6\flower_data\flower_photos"
    tfrecord_path = r"G:\tensorflow_google\chapter_6\flower_data1.tfrecords"
    image_to_tfrecord(image_path, tfrecord_path)

if __name__ == '__main__':
    main()
