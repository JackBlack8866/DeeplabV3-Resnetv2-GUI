import os
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置模型和输入图片的路径
MODEL_PATH = 'frozen_inference_graph3.onnx'  # 替换为您的模型路径
IMAGE_PATH = 'frankfurt_000000_001751_leftImg8bit.png'  # 替换为您的输入图片路径


# 载入 ONNX 模型
def load_model(model_path):
    model = onnx.load(model_path)
    return model


# 使用 ONNX Runtime 进行推断
def predict(image, model_path):
    ort_session = ort.InferenceSession(model_path)

    # 调整图片大小并预处理
    input_size = (2049, 1025)
    resized_image = image.convert('RGB').resize(input_size, Image.BILINEAR)
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)  # 增加 batch 维度
    image_for_prediction = image_for_prediction / 127.5 - 1  # 归一化处理

    # 获取输入和输出名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 进行推断
    raw_prediction = ort_session.run([output_name], {input_name: image_for_prediction})[0]

    # 打印原始预测形状
    print("Raw prediction shape:", raw_prediction.shape)

    # 后处理：将原始输出转换为分割输出
    seg_map = np.argmax(raw_prediction, axis=3)  # 在类别维度上取 argmax
    return np.squeeze(seg_map).astype(np.int8)  # 移除多余维度并转换类型


# 加载并处理输入图片
image = Image.open(IMAGE_PATH)

# 进行推断
seg_map = predict(image, MODEL_PATH)


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
    # # 保存分割结果为单独的图像
    # segmented_image = Image.fromarray(seg_image)
    # segmented_image.save(os.path.join('onnx_segmentation_map.png'))
    plt.axis('off')
    plt.title('Segmentation Map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('Segmentation Overlay')

    plt.show()


# 调用可视化函数
vis_segmentation(image, seg_map)




