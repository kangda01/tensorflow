import os
import random, shutil

def partitioned_dataset(file_Dir, target_Dir):
    #sonDirPath = []
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
                    old_dir = os.path.join(son_dir_name, name)
                    new_dir = os.path.join(target_Dir, set_name_str, dir )
                    isExists = os.path.exists(new_dir)
                    if not isExists:
                        os.makedirs(new_dir)
                    # newtarget_Dir = os.path.join(new_dir, name)
                    shutil.move(old_dir, new_dir)


def main():
    file_Dir = r"G:\tensorflow_google\chapter_6\flower_data\flower_photos" #源图片文件夹路径
    target_Dir = r'C:\Users\kangda\Desktop\flower_photos'       #目标路径
    partitioned_dataset(file_Dir, target_Dir)


if __name__ == '__main__':
    main()
