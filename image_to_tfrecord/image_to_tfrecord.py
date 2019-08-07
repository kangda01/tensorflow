
from pathlib import Path
import tensorflow as tf
from PIL import Image
import math


'''get_images_dir函数为了得到每个图片地址和对应的标签，同时得到总的图片个数。
    return:
        all_image_paths_labels:得到每个图片地址和对应的标签,以便后续读取存储
        image_count：为了后续显示处理总进度
'''
def get_images_dir(image_path):
    data_root = Path(image_path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    image_count = len(all_image_paths)
    # 确定每种图像的标签
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    # 为每个标签分配索引
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    # 创建每个图片的标签索引
    all_image_labels = [label_to_index[Path(path).parent.name]
                        for path in all_image_paths]
    all_image_paths_labels = zip(all_image_paths, all_image_labels)
    return all_image_paths_labels, image_count

#@profile
def image_to_tfrecord(set_name, image_path, output_path):

    all_image_paths_labels, image_count = get_images_dir(image_path)
    #实现向上取整,得到总的tfrecord文件个数
    tfrecords_files_numbers = math.ceil(image_count / 1000)
    number = 0
    #处理图片数据
    for i, file_name in enumerate(all_image_paths_labels):
        if not i % 1000:
            number += 1
            tfrecord_path = Path(output_path).joinpath(
                ''.join([set_name, ('-%.3d-of-%.3d.tfrecords' % (number, tfrecords_files_numbers))]))
            writer = tf.python_io.TFRecordWriter(str(tfrecord_path))
        #try...except 添加报错机制，防止图片损坏转换不成功
        try:
            #添加显示进度信息
            print('\r>> Converting image  %d/%d' % (i+1, image_count),end="")
            #采用PIL模块下的Image读取图片
            image_raw_data = Image.open(file_name[0])
            #获取图片的长宽高，保存在tftecort中，以便还原图片信息使用
            height = image_raw_data.height
            width = image_raw_data.width
            channels = image_raw_data.layers
            #转换成二进制格式
            image_raw = image_raw_data.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_frature(file_name[1]),
                'image_raw': _bytes_feature(image_raw),
                'height': _int64_frature(height),
                'width': _int64_frature(width),
                'channels': _int64_frature(channels)}))
            writer.write(example.SerializeToString())
        except IOError as shibai:
            print('Wrong: ' + file_name)
            print('Error: ', shibai)
            print('Skip it\n')
    writer.close()


# 生成整数型的属性
def _int64_frature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():

    image_path = r"G:\tensorflow_google\chapter_6\flower_data\flower_photos"
    output_path = r"G:\tensorflow_google\chapter_6\flower_data"
    image_to_tfrecord("training", image_path, output_path)

if __name__ == '__main__':
    main()
    
