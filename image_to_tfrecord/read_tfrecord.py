import tensorflow as tf
import  glob
from PIL import Image
import os.path

train_files = glob.glob(r"G:\tensorflow_google\chapter_6\flower_data\training-*")
test_files = glob.glob(r"G:\tensorflow_google\chapter_6\flower_data\testing-*")
verification_files = glob.glob(r"G:\tensorflow_google\chapter_6\flower_data\verification-*")
def parser(record):
    features = tf.parse_single_example(record,
                    features={'label': tf.FixedLenFeature([], tf.int64),
                              'height': tf.FixedLenFeature([], tf.int64),
                              'width': tf.FixedLenFeature([], tf.int64),
                              'channels': tf.FixedLenFeature([], tf.int64),
                            'image_raw': tf.FixedLenFeature([], tf.string)})
    #从原始图像解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    #将图片的unit8格式转换成float32
    retyped_images = tf.cast(decoded_image, tf.float32)
    # 这个版本的tensorflow的reshape不支持tf.int64,所以转换成tf.int32
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)
    image = tf.reshape(retyped_images,
                               shape=[height, width, channels])
    label = tf.cast(features['label'],tf.int32)
    return image, label


#将图片尺寸调整一致，大小和神经网络输入一样
def preprocess_image(image, height, width):
  image = tf.image.resize(image, [height, width])
  image /= 255.0  # normalize to [0,1] range
  return image

data_dir = tf.placeholder(tf.string)
image_height = tf.placeholder(tf.int32)
image_width = tf.placeholder(tf.int32)
dataset = tf.data.TFRecordDataset(data_dir)
dataset = dataset.map(parser)
dataset = dataset.map(lambda image, label :
                      (preprocess_image(image, image_height, image_width), label))
# dataset = dataset.shuffle(buffer_size=10000).batch(100)
iterator = dataset.make_initializable_iterator()
image, label = iterator.get_next()


with tf.Session() as sess:
    sess.run(iterator.initializer,
             feed_dict={data_dir:train_files, image_height:299, image_width:299})
    pass
