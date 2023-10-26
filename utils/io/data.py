import os
import cv2
import json
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt


class DataGen:

    def __init__(self, path, split_ratio, x, y, color_space='rgb'):
        self.x = x
        self.y = y
        self.path = path
        self.color_space = color_space
        self.path_train_images = path + "train/images/"
        self.path_train_labels = path + "train/labels/"
        self.path_test_images = path + "test/images/"
        self.path_test_labels = path + "test/labels/"
        self.image_file_list = get_png_filename_list(self.path_train_images)
        self.label_file_list = get_png_filename_list(self.path_train_labels)
        self.image_file_list[:], self.label_file_list[:] = self.shuffle_image_label_lists_together()
        self.split_index = int(split_ratio * len(self.image_file_list))
        self.x_train_file_list = self.image_file_list[self.split_index:]
        self.y_train_file_list = self.label_file_list[self.split_index:]
        self.x_val_file_list = self.image_file_list[:self.split_index]
        self.y_val_file_list = self.label_file_list[:self.split_index]
        self.x_test_file_list = get_png_filename_list(self.path_test_images)
        self.y_test_file_list = get_png_filename_list(self.path_test_labels)

    def generate_data(self, batch_size, train=False, val=False, test=False):
        """Replaces Keras' native ImageDataGenerator."""
        try:
            if train is True:
                image_file_list = self.x_train_file_list
                label_file_list = self.y_train_file_list
            elif val is True:
                image_file_list = self.x_val_file_list
                label_file_list = self.y_val_file_list
            elif test is True:
                image_file_list = self.x_test_file_list
                label_file_list = self.y_test_file_list
        except ValueError:
            print('one of train or val or test need to be True')

        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(self.x_train_file_list):
                    i = 0
                if i < len(image_file_list):
                    sample_image_filename = image_file_list[i]
                    sample_label_filename = label_file_list[i]
                    # print('image: ', image_file_list[i])
                    # print('label: ', label_file_list[i])
                    if train or val:
                        image = cv2.imread(self.path_train_images + sample_image_filename, 1)
                        label = cv2.imread(self.path_train_labels + sample_label_filename, 0)
                    elif test is True:
                        image = cv2.imread(self.path_test_images + sample_image_filename, 1)
                        label = cv2.imread(self.path_test_labels + sample_label_filename, 0)
                    # image, label = self.change_color_space(image, label, self.color_space)
                    label = np.expand_dims(label, axis=2)
                    if image.shape[0] == self.x and image.shape[1] == self.y:
                        image_batch.append(image.astype("float32"))
                    else:
                        print('the input image shape is not {}x{}'.format(self.x, self.y))
                        
                    if label.shape[0] == self.x and label.shape[1] == self.y:
                        label_batch.append(label.astype("float32"))
                    else:
                        print('the input label shape is not {}x{}'.format(self.x, self.y))
                i += 1
            if image_batch and label_batch:
                image_batch = normalize(np.array(image_batch))
                label_batch = normalize(np.array(label_batch))
                yield (image_batch, label_batch)

    def get_num_data_points(self, train=False, val=False):
        try:
            image_file_list = self.x_train_file_list if val is False and train is True else self.x_val_file_list
        except ValueError:
            print('one of train or val need to be True')

        return len(image_file_list)

    def shuffle_image_label_lists_together(self):
        combined = list(zip(self.image_file_list, self.label_file_list))
        random.shuffle(combined)
        return zip(*combined)

    @staticmethod
    def change_color_space(image, label, color_space):
        if color_space.lower() is 'hsi' or 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
        elif color_space.lower() is 'lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)
        return image, label
def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


def get_png_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".png" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def get_jpg_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".jpg" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def load_jpg_images(path):
    file_list = get_jpg_filename_list(path)
    temp_list = []
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    # x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return temp_list, file_list


def load_png_images(path, target_x, target_y):

    temp_list = []
    file_list = get_png_filename_list(path)
    for filename in file_list:
        img = cv2.imread(path + filename, 1)

        # 如果图像的尺寸与目标尺寸不同，则调整大小
        if img.shape[0] != target_x or img.shape[1] != target_y:
            ratio = min(target_x / img.shape[0], target_y / img.shape[1])
            new_x = int(img.shape[1] * ratio)
            new_y = int(img.shape[0] * ratio)
            img = cv2.resize(img, (new_x, new_y))
            
            # 如果需要，你还可以将图像放在中心，这样它的周围就会有一些填充
            delta_w = target_x - new_x
            delta_h = target_y - new_y
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    #temp_list = np.reshape(temp_list,(temp_list.shape[0], temp_list.shape[1], temp_list.shape[2], 3))
    return temp_list, file_list


def load_data(path, target_x, target_y):
    # path_train_images = path + "train/images/padded/"
    # path_train_labels = path + "train/labels/padded/"
    # path_test_images = path + "test/images/padded/"
    # path_test_labels = path + "test/labels/padded/"
    path_train_images = path + "train/images/"
    path_train_labels = path + "train/labels/"
    path_test_images = path + "test/images/"
    path_test_labels = path + "test/labels/"
    x_train, train_image_filenames_list = load_png_images(path_train_images, target_x, target_y)
    y_train, train_label_filenames_list = load_png_images(path_train_labels, target_x, target_y)
    x_test, test_image_filenames_list = load_png_images(path_test_images, target_x, target_y)
    y_test, test_label_filenames_list = load_png_images(path_test_labels, target_x, target_y)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    x_test = normalize(x_test)
    y_test = normalize(y_test)
    return x_train, y_train, x_test, y_test, test_label_filenames_list


def load_test_images(path, target_x, target_y):
    path_test_images = path + "test/images/"
    x_test, test_image_filenames_list = load_png_images(path_test_images, target_x, target_y)
    x_test = normalize(x_test)
    return x_test, test_image_filenames_list


def save_results(np_array, color_space, outpath, test_label_filenames_list):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        pred = np_array[i]
        # if color_space.lower() is 'hsi' or 'hsv':
        #     pred = cv2.cvtColor(pred, cv2.COLOR_HSV2RGB)
        # elif color_space.lower() is 'lab':
        #     pred = cv2.cvtColor(pred, cv2.COLOR_Lab2RGB)
        cv2.imwrite(outpath + filename, pred * 255.)
        i += 1


def save_rgb_results(np_array, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        cv2.imwrite(outpath + filename, np_array[i] * 255.)
        i += 1


def save_history(model, model_name, training_history, dataset, n_filters, epoch, learning_rate, loss,
                 color_space, path=None, temp_name=None):
    formatted_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_weight_filename = temp_name if temp_name else formatted_date

    fileFullName = os.path.join(path, '{}.hdf5'.format(save_weight_filename))

    model.save(fileFullName)
    with open(os.path.join(path, '{}.json'.format(save_weight_filename)), 'w') as f:
        json.dump(training_history.history, f, indent=2)

    json_list = [os.path.join(path, '{}.json'.format(save_weight_filename))]
    for json_filename in json_list:
        with open(json_filename) as f:
            # convert the loss json object to a python dict
            loss_dict = json.load(f)
        print_list = ['loss', 'val_loss', 'dice_coef', 'val_dice_coef']
        for item in print_list:
            item_list = []
            if item in loss_dict:
                item_list.extend(loss_dict.get(item))
                plt.plot(item_list)
        plt.title('model:{} lr:{} epoch:{} #filtr:{} Colorspaces:{}'.format(model_name, learning_rate,
                                                                            epoch, n_filters, color_space))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'test_loss', 'train_dice', 'test_dice'], loc='upper left')
        plt.savefig('{}{}.png'.format(path, save_weight_filename))
        plt.show()
        plt.clf()



