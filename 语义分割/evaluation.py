

import os
import numpy as np
from PIL import Image

"""
保存灰度图
"""

# # 映射字典
# trainId_to_id = {
#     0: 7,
#     1: 8,
#     2: 11,
#     3: 12,
#     4: 13,
#     5: 17,
#     6: 19,
#     7: 20,
#     8: 21,
#     9: 22,
#     10: 23,
#     11: 24,
#     12: 25,
#     13: 26,
#     14: 27,
#     15: 28,
#     16: 31,
#     17: 32,
#     18: 33,
#     # 19: 0
# }
#
# # ID 映射函数
# def map_ids(image, mapping):
#     mapped_image = np.zeros_like(image)
#     for original_id, new_id in mapping.items():
#         mapped_image[image == original_id] = new_id
#     return mapped_image
#
# # 计算像素准确率并保存图像
# def calculate_pixel_accuracy(prediction_path, ground_truth_path, output_path, mapping):
#     total_pixels = 0
#     correct_pixels = 0
#     results = []
#
#     # 确保输出文件夹存在
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     for pred_file in os.listdir(prediction_path):
#         if pred_file.endswith(('.png', '.jpg', '.jpeg')):
#             pred_img = Image.open(os.path.join(prediction_path, pred_file)).convert('L')  # 转为灰度图
#             gt_img = Image.open(os.path.join(ground_truth_path, pred_file)).convert('L')  # 转为灰度图
#
#             # 将图像转换为数组并映射类别ID
#             pred_array = map_ids(np.array(pred_img), mapping).flatten()
#             gt_array = map_ids(np.array(gt_img), mapping).flatten()
#
#             total_pixels += gt_array.size
#             correct_pixels += np.sum(pred_array == gt_array)
#
#             # 计算当前图像的像素准确率
#             image_accuracy = np.sum(pred_array == gt_array) / gt_array.size if gt_array.size > 0 else 0
#             results.append(image_accuracy)
#
#             print(f'图像 {pred_file} 像素准确率: {image_accuracy:.4f}')
#
#             # 保存预测和真实图像
#             pred_img.save(os.path.join(output_path, f'pred_{pred_file}'))
#             gt_img.save(os.path.join(output_path, f'gt_{pred_file}'))
#
#     average_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
#     return average_accuracy, results
#
# # 使用示例
# prediction_path = './predict/predict_frankfurt_5'
# ground_truth_path = './label/frankfurt_5'
# output_path = './gray_images/'  # 保存结果的文件夹
#
# average_pixel_accuracy, individual_accuracies = calculate_pixel_accuracy(prediction_path, ground_truth_path, output_path, trainId_to_id)
#
# print(f'平均像素准确率: {average_pixel_accuracy:.4f}')





"""
去掉映射
"""
# import os
# import numpy as np
# from PIL import Image
#
# # 计算像素准确率并保存图像
# def calculate_pixel_accuracy(prediction_path, ground_truth_path, output_path):
#     total_pixels = 0
#     correct_pixels = 0
#     results = []
#
#     # 确保输出文件夹存在
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     for pred_file in os.listdir(prediction_path):
#         if pred_file.endswith(('.png', '.jpg', '.jpeg')):
#             pred_img = Image.open(os.path.join(prediction_path, pred_file)).convert('L')  # 转为灰度图
#             gt_img = Image.open(os.path.join(ground_truth_path, pred_file)).convert('L')  # 转为灰度图
#
#             # 将图像转换为数组
#             pred_array = np.array(pred_img).flatten()
#             gt_array = np.array(gt_img).flatten()
#
#             total_pixels += gt_array.size
#             correct_pixels += np.sum(pred_array == gt_array)
#
#             # 计算当前图像的像素准确率
#             image_accuracy = np.sum(pred_array == gt_array) / gt_array.size if gt_array.size > 0 else 0
#             results.append(image_accuracy)
#
#             print(f'图像 {pred_file} 像素准确率: {image_accuracy:.4f}')
#
#             # 保存预测和真实图像
#             pred_img.save(os.path.join(output_path, f'pred_{pred_file}'))
#             gt_img.save(os.path.join(output_path, f'gt_{pred_file}'))
#
#     average_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
#     return average_accuracy, results
#
# # 使用示例
# prediction_path = './predict/predict_frankfurt_5'
# ground_truth_path = './label/frankfurt_5'
# output_path = './evaluation_results/'  # 保存结果的文件夹
#
# average_pixel_accuracy, individual_accuracies = calculate_pixel_accuracy(prediction_path, ground_truth_path, output_path)
#
# print(f'平均像素准确率: {average_pixel_accuracy:.4f}')


"""

"""

# 映射字典
trainId_to_id = {
    0: 7,
    1: 8,
    2: 11,
    3: 12,
    4: 13,
    5: 17,
    6: 19,
    7: 20,
    8: 21,
    9: 22,
    10: 23,
    11: 24,
    12: 25,
    13: 26,
    14: 27,
    15: 28,
    16: 31,
    17: 32,
    18: 33,
    19: 0
}

# ID 映射函数
def map_ids(image, mapping):
    mapped_image = np.zeros_like(image)
    for original_id, new_id in mapping.items():
        mapped_image[image == original_id] = new_id
    return mapped_image

# 计算像素准确率
def calculate_pixel_accuracy(prediction_path, ground_truth_path, mapping):
    total_pixels = 0
    correct_pixels = 0
    results = []

    for pred_file in os.listdir(prediction_path):
        if pred_file.endswith(('.png', '.jpg', '.jpeg')):
            pred_img = Image.open(os.path.join(prediction_path, pred_file)).convert('L')  # 转为灰度图
            gt_img = Image.open(os.path.join(ground_truth_path, pred_file)).convert('L')  # 转为灰度图

            # 将图像转换为数组并映射类别ID
            pred_array = map_ids(np.array(pred_img), mapping).flatten()
            gt_array = map_ids(np.array(gt_img), mapping).flatten()

            total_pixels += gt_array.size
            correct_pixels += np.sum(pred_array == gt_array)

            # 计算当前图像的像素准确率
            image_accuracy = np.sum(pred_array == gt_array) / gt_array.size if gt_array.size > 0 else 0
            results.append(image_accuracy)

            print(f'图像 {pred_file} 像素准确率: {image_accuracy:.4f}')

    average_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    return average_accuracy, results

# 使用示例
prediction_path = './predict/predict_frankfurt_5'
ground_truth_path = './label/frankfurt_5'

average_pixel_accuracy, individual_accuracies = calculate_pixel_accuracy(prediction_path, ground_truth_path, trainId_to_id)

print(f'平均像素准确率: {average_pixel_accuracy:.4f}')

