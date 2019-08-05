import tensorflow as tf

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

data_dir = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(data_dir)
dataset = dataset.map(parser)
iterator = dataset.make_initializable_iterator()
image, label = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer,
             feed_dict={data_dir:r"C:\Users\kangda\Desktop\flower.tfrecords"})

    # for i in range(10):
    #     print(i, sess.run([image, label]), image.shape)

    # i = 0
    # while 1:
    #     i += 1
    #     try:
    #         print(i, sess.run([image, label]))
    #     except tf.errors.OutOfRangeError:
    #         print("end")
    #     break
