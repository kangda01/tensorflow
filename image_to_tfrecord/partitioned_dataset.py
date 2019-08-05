import os
import glob
import random, shutil


'''注意处理前请注意备份文件 '''
#定义这个函数是计算总的图片个数，为了显示处理进度
def countFile(image_path):
    sub_dirs = [x[0] for x in os.walk(image_path)][1:]
    #读取所有的子目录
    file_list = []
    for label, sub_dir in enumerate(sub_dirs):
        extensions = ['jpg', 'jpeg']
        #os.path.basename函数返回文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(image_path, dir_name, '*.'+extension)
            #glob.glob(pathname)返回所有匹配的文件路径列表。它只有一个参数pathname，
            # 定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径
            #注意extend和append的区别
            file_list.extend(glob.glob(file_glob))
        #如果file_list为空，跳出本次循环
        if not file_list: continue
    number = len(file_list)
    return number


def partitioned_dataset(file_Dir, target_Dir):
    #sonDirPath = []
    total_image_num = countFile(file_Dir)
    i = 0
    all_dir = os.listdir(file_Dir)  # 列出指定路径下的全部文件夹，以列表的方式保存
    for dir in all_dir:  # 遍历指定路径下的全部文件和文件夹
        son_dir_name = os.path.join(file_Dir, dir)  # 子文件夹的路径名称
        if os.path.isdir(son_dir_name):
            path_dir = os.listdir(son_dir_name)  # 取图片的原始路径
            file_number = len(path_dir)
            validation_rate = 0.10  # 
            testing_rate = 0.10
            validation_number = int(file_number * validation_rate)
            testing_number = int(file_number * testing_rate) 
            training_number = file_number - validation_number - testing_number
            random.shuffle(path_dir)
            training_image_names = path_dir[:training_number]
            testing_image_names = path_dir[training_number:testing_number+training_number]
            validation_image_names = path_dir[-validation_number:]
            
            
            for set_name, set_name_str in [(training_image_names, "training"),
                                           (testing_image_names, "testing"),
                                    (validation_image_names, "validation")]:
                for name in set_name:
                    i += 1
                    old_dir = os.path.join(son_dir_name, name)
                    new_dir = os.path.join(target_Dir, set_name_str, dir )
                    isExists = os.path.exists(new_dir)
                    if not isExists:
                        os.makedirs(new_dir)
                    # newtarget_Dir = os.path.join(new_dir, name)
                    shutil.move(old_dir, new_dir)
                    print('\r>> 处理进度： %d/%d' % (i, total_image_num), end="")


def main():
    file_Dir = r"G:\tensorflow_google\chapter_6\flower_data\flower_photos" #源图片文件夹路径
    target_Dir = r'C:\Users\kangda\Desktop\flower_photos'       #目标路径
    partitioned_dataset(file_Dir, target_Dir)



if __name__ == '__main__':
    main()
