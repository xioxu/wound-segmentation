import time
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from models.unets import Unet2D
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from models.FCN import FCN_Vgg16_16s

from utils.learning.metrics import dice_coef, precision, recall
from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.data import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGen


# settings
input_dim_x = 512
input_dim_y = 512
color_space = 'rgb'

#weight_file_name = '2019-12-19 01%3A53%3A15.480800.hdf5'
weight_file_name = '20231026_130624.hdf5'


# 加载模型
with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D, 'BilinearUpsampling': BilinearUpsampling}):
    model = load_model('./training_history/' + weight_file_name,
                       custom_objects={'recall': recall,
                                       'precision': precision,
                                       'dice_coef': dice_coef})

def resize_and_pad(image, target_width, target_height):
    # First, resize the image while keeping the aspect ratio
    img_resized = resize_keep_aspect_ratio(image, target_width, target_height)

    # Then, pad the resized image to match the target dimensions
    delta_w = target_width - img_resized.shape[1]
    delta_h = target_height - img_resized.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Black padding
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img_padded

def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr

def resize_keep_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    if h != target_height or w != target_width:
        aspect_ratio = w / h
        if w > h:
            new_w = target_width
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_height
            new_w = int(new_h * aspect_ratio)
        
        return cv2.resize(image, (new_w, new_h))
    return image


def ccl(prediction_mask):
    # 使用cv2.connectedComponentsWithStats获取连通分量及其统计信息
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((prediction_mask > 0).astype('uint8'))

    # 定义面积阈值
    # 根据像素总数动态确定面积阈值
    total_wound_pixels = prediction_mask.size
    area_threshold_total = 0.00135 * total_wound_pixels 

    # 根据所有连通区域的总像素来动态确定面积阈值
    total_connected_pixels = np.sum(stats[1:, cv2.CC_STAT_AREA])  # 跳过背景的像素
    area_threshold = 0.08 * total_connected_pixels  # 例如, 所有连通区域的像素总数的1%


    # 创建一个空的掩膜来保存过滤后的结果
    filtered_mask = np.zeros_like(prediction_mask)

    # 遍历每个连通分量，保留面积大于阈值的分量
    for i in range(1, num_labels):  # 跳过标签0，因为它是背景
        if stats[i, cv2.CC_STAT_AREA] > area_threshold and stats[i, cv2.CC_STAT_AREA] > area_threshold_total:
            filtered_mask[labels_im == i] = prediction_mask[labels_im == i]
    return filtered_mask

def save_prediction_mask(original_path, prediction):
    # Load the original image
    original_image = cv2.imread(original_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert color space

    # Calculate padding added during preprocessing
    delta_w = input_dim_x - original_image.shape[1]
    delta_h = input_dim_y - original_image.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Assuming ccl is a previously defined function to process the prediction
    prediction[0] = ccl(prediction[0])
    
    # Remove padding from prediction
    prediction_unpadded = prediction[0, top:-bottom, left:-right] if bottom != 0 and right != 0 else prediction[0, top:, left:]
    
    # Resize the unpadded prediction to match the original image size
    prediction_mask = cv2.resize(prediction_unpadded * 255, (original_image.shape[1], original_image.shape[0]))
    
    # Convert the prediction mask to binary (assuming 127.5 as threshold)
    prediction_mask = (prediction_mask > 127.5).astype(np.uint8) * 255

    save_dir = os.path.join(os.path.dirname(original_path), "predicted")
    base_name = os.path.basename(original_path)  # e.g., "1.jpg"
    file_name_without_ext = os.path.splitext(base_name)[0]  # e.g., "1"
    predicted_file_name = f"{file_name_without_ext}_predicted.jpg"
    predicted_file_path = os.path.join(save_dir, predicted_file_name)
    
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(predicted_file_path, prediction_mask)

def save_overlayed_results(original_path, prediction):
    # Load the original image
    original_image = cv2.imread(original_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert color space

    # Calculate padding added during preprocessing
    delta_w = input_dim_x - original_image.shape[1]
    delta_h = input_dim_y - original_image.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    prediction[0] = ccl(prediction[0])
    # Remove padding from prediction
    prediction_unpadded = prediction[0, top:-bottom, left:-right] if bottom != 0 and right != 0 else prediction[0, top:, left:]
    
    
    
    # Resize the unpadded prediction to match the original image size
    prediction_mask = cv2.resize(prediction_unpadded * 255, (original_image.shape[1], original_image.shape[0]))
    
    
    #没有处理padding 但是结果看起来也是正确的
    #prediction_mask = cv2.resize(prediction[0] * 255, (original_image.shape[1], original_image.shape[0])) 

 
    # Create a three-channel mask
    mask_overlay = np.zeros_like(original_image)
    mask_overlay[prediction_mask > 127] = [0, 0, 255]  # Use blue color for the predicted area

    # Merge the original image and the mask using weighted blending
    alpha = 0.4  # Transparency for the overlay (40%)
    overlayed_image = cv2.addWeighted(original_image, 1, mask_overlay, alpha, 0)

    save_dir = os.path.join(os.path.dirname(original_path),"predicted")
    base_name = os.path.basename(original_path)  # e.g., "1.jpg"
    file_name_without_ext = os.path.splitext(base_name)[0]  # e.g., "1"
    predicted_file_name = f"{file_name_without_ext}_predicted.jpg"
    predicted_file_path = os.path.join(save_dir, predicted_file_name)
 
    cv2.imwrite(predicted_file_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

def process_image(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = resize_and_pad(img, input_dim_x, input_dim_y)
    img = img.astype("float32")
    img = normalize(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=1)
    save_overlayed_results(imgPath, prediction)

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
        return result
    return wrapper

@timer_decorator
def process_directory2(directory_path):
    # List all files in the directory
    files_in_directory = os.listdir(directory_path)
    
    # Filter for .png and .jpg files
    image_files = [f for f in files_in_directory if os.path.splitext(f)[1] in ['.png', '.jpg', '.JPG', '.PNG']]
    
    # Process each image file
    for image_file in image_files:
        process_image(os.path.join(directory_path, image_file))

def prepare_image(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = resize_and_pad(img, input_dim_x, input_dim_y)
    img = img.astype("float32")
    img = normalize(img)
    return img

@timer_decorator
def process_directory(directory_path):
    # List all files in the directory
    files_in_directory = os.listdir(directory_path)
    
    # Filter for .png and .jpg files
    image_files = [f for f in files_in_directory if os.path.splitext(f)[1] in ['.png', '.jpg', '.JPG', '.PNG']]
    
    # Prepare all images
    images = [prepare_image(os.path.join(directory_path, image_file)) for image_file in image_files]
    
    # Convert list of images to a batch (numpy array)
    image_batch = np.stack(images, axis=0)
    
    # Batch prediction
    predictions = model.predict(image_batch, verbose=1)
    
    # Process each predicted result
    for imgPath, prediction in zip(image_files, predictions):
        save_overlayed_results(os.path.join(directory_path, imgPath), prediction[np.newaxis, ...])
        #save_prediction_mask(os.path.join(directory_path, imgPath), prediction[np.newaxis, ...])

directory_path = "./data/Medetec_foot_ulcer_224/test/1"

process_directory(directory_path)

