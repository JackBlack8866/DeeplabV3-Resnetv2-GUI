'''
单张
'''

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置模型和输入图片的路径
MODEL_PATH = 'frozen_inference_graph.pb'  # 替换为您的模型路径
IMAGE_PATH = 'frankfurt_000000_000294_leftImg8bit.png'  # 替换为您的输入图片路径

# 加载 PB 模型
def load_model(model_path):
    model = tf.Graph()
    with model.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    return model

# 载入模型
model = load_model(MODEL_PATH)

# 取得输入和输出张量
with model.as_default():
    input_tensor = model.get_tensor_by_name('sub_7:0')  # 根据您使用的模型调整输入张量名称
    output_tensor = model.get_tensor_by_name('ResizeBilinear_2:0')  # 根据您使用的模型调整输出张量名称

# 加载并预处理输入图片
image = Image.open(IMAGE_PATH)
input_size = (2049, 1025)
old_size = image.size
desired_ratio = input_size[0] / input_size[1]
old_ratio = old_size[0] / old_size[1]

if old_ratio < desired_ratio:  # '<': 裁剪, '>': 填充
    new_size = (old_size[0], int(old_size[0] / desired_ratio))
else:
    new_size = (int(old_size[1] * desired_ratio), old_size[1])

delta_w = new_size[0] - old_size[0]
delta_h = new_size[1] - old_size[1]
padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
cropped_image = ImageOps.expand(image, padding)

# 调整大小并转换为数组
resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)
image_for_prediction = np.asarray(resized_image).astype(np.float32)
image_for_prediction = np.expand_dims(image_for_prediction, 0)
image_for_prediction = image_for_prediction / 127.5 - 1

# 进行推断
with tf.compat.v1.Session(graph=model) as sess:
    raw_prediction = sess.run(output_tensor, feed_dict={input_tensor: image_for_prediction})

    # 后处理：将原始输出转换为分割输出
    width, height = cropped_image.size
    resized_prediction = tf.image.resize(raw_prediction, (height, width))
    seg_map = tf.argmax(resized_prediction, axis=3)

    # 使用 sess.run() 来获取 seg_map 的值
    seg_map = sess.run(tf.squeeze(seg_map))
    seg_map = seg_map.astype(np.int8)  # 先获取 numpy 数组再进行转换

# 可视化分割结果
def create_cityscapes_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap

def label_to_color_image(label):
    colormap = create_cityscapes_label_colormap()
    return colormap[label]

def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('Segmentation Map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('Segmentation Overlay')

    plt.show()

vis_segmentation(cropped_image, seg_map)

'''
文件夹
'''

# import os
# import numpy as np
# import tensorflow as tf
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import time
#
# # 设置模型路径和输入/输出图片文件夹路径
# MODEL_PATH = 'frozen_inference_graph.pb'  # 模型路径
# INPUT_FOLDER = './images/frankfurt6/'  # 输入图片文件夹
# OUTPUT_FOLDER = './predict/predict_frankfurt6/'  # 输出图片文件夹
#
# # 确保输出文件夹存在
# if not os.path.exists(OUTPUT_FOLDER):
#     os.makedirs(OUTPUT_FOLDER)
#
# # 加载 PB 模型
# def load_model(model_path):
#     model = tf.Graph()
#     with model.as_default():
#         graph_def = tf.compat.v1.GraphDef()
#         with tf.io.gfile.GFile(model_path, 'rb') as f:
#             graph_def.ParseFromString(f.read())
#             tf.import_graph_def(graph_def, name='')
#     return model
#
# # 载入模型
# model = load_model(MODEL_PATH)
#
# # 取得输入和输出张量
# with model.as_default():
#     input_tensor = model.get_tensor_by_name('sub_7:0')  # 根据模型调整输入张量名称
#     output_tensor = model.get_tensor_by_name('ResizeBilinear_2:0')  # 根据模型调整输出张量名称
#
# # 创建分割颜色映射
# def create_cityscapes_label_colormap():
#     colormap = np.zeros((256, 3), dtype=np.uint8)
#     colormap[0] = [128, 64, 128]
#     colormap[1] = [244, 35, 232]
#     colormap[2] = [70, 70, 70]
#     colormap[3] = [102, 102, 156]
#     colormap[4] = [190, 153, 153]
#     colormap[5] = [153, 153, 153]
#     colormap[6] = [250, 170, 30]
#     colormap[7] = [220, 220, 0]
#     colormap[8] = [107, 142, 35]
#     colormap[9] = [152, 251, 152]
#     colormap[10] = [70, 130, 180]
#     colormap[11] = [220, 20, 60]
#     colormap[12] = [255, 0, 0]
#     colormap[13] = [0, 0, 142]
#     colormap[14] = [0, 0, 70]
#     colormap[15] = [0, 60, 100]
#     colormap[16] = [0, 80, 100]
#     colormap[17] = [0, 0, 230]
#     colormap[18] = [119, 11, 32]
#     return colormap
#
# def label_to_color_image(label):
#     colormap = create_cityscapes_label_colormap()
#     return colormap[label]
#
# # 处理并保存分割结果
# def process_and_save_image(image_path, output_path):
#     # 加载并预处理输入图片
#     image = Image.open(image_path)
#     input_size = (2049, 1025)
#     old_size = image.size
#     desired_ratio = input_size[0] / input_size[1]
#     old_ratio = old_size[0] / old_size[1]
#
#     if old_ratio < desired_ratio:  # '<': 裁剪, '>': 填充
#         new_size = (old_size[0], int(old_size[0] / desired_ratio))
#     else:
#         new_size = (int(old_size[1] * desired_ratio), old_size[1])
#
#     delta_w = new_size[0] - old_size[0]
#     delta_h = new_size[1] - old_size[1]
#     padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
#     cropped_image = ImageOps.expand(image, padding)
#
#     # 调整大小并转换为数组
#     resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)
#     image_for_prediction = np.asarray(resized_image).astype(np.float32)
#     image_for_prediction = np.expand_dims(image_for_prediction, 0)
#     image_for_prediction = image_for_prediction / 127.5 - 1
#
#     # 进行推断并计算时间
#     start_time = time.time()
#     with tf.compat.v1.Session(graph=model) as sess:
#         raw_prediction = sess.run(output_tensor, feed_dict={input_tensor: image_for_prediction})
#
#         # 后处理：将原始输出转换为分割输出
#         width, height = cropped_image.size
#         resized_prediction = tf.image.resize(raw_prediction, (height, width))
#         seg_map = tf.argmax(resized_prediction, axis=3)
#
#         # 使用 sess.run() 来获取 seg_map 的值
#         seg_map = sess.run(tf.squeeze(seg_map))
#         seg_map = seg_map.astype(np.int8)  # 先获取 numpy 数组再进行转换
#
#         # resize回2048*1024
#         seg_map= Image.fromarray(seg_map).resize((2048, 1024), Image.NEAREST)
#
#     end_time = time.time()
#     prediction_time = end_time - start_time
#
#     # 保存分割结果
#     seg_image = label_to_color_image(seg_map).astype(np.uint8)
#     Image.fromarray(seg_image).save(output_path)
#
#     return seg_map, prediction_time
#
# # 处理文件夹中的每张图片并记录时间
# total_time = 0
# num_images = 0
#
# for image_file in os.listdir(INPUT_FOLDER):
#     if image_file.endswith(('.png', '.jpg', '.jpeg')):
#         input_image_path = os.path.join(INPUT_FOLDER, image_file)
#         output_image_path = os.path.join(OUTPUT_FOLDER, image_file)
#
#         print(f'正在处理 {input_image_path}')
#         seg_map, prediction_time = process_and_save_image(input_image_path, output_image_path)
#         total_time += prediction_time
#         num_images += 1
#
#         print(f'已保存到 {output_image_path}')
#         print(f'预测时间: {prediction_time:.2f} 秒')
#
# # 计算平均时间
# average_time = total_time / num_images if num_images > 0 else 0
# print(f'总处理图像数量: {num_images}')
# print(f'总时间: {total_time:.2f} 秒')
# print(f'平均时间: {average_time:.2f} 秒')



