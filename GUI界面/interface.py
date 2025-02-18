import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Label
import onnxruntime as ort
from PIL import Image, ImageTk
import csv
import os
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from model import build_model

# 读取分类信息
def load_scene_classes():
    try:
        with open('scene_classes.csv', encoding='utf-8') as file:
            reader = csv.reader(file)
            scene_classes_list = list(reader)
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}. Trying alternative encoding...")
        with open('scene_classes.csv', encoding='gb18030') as file:
            reader = csv.reader(file)
            scene_classes_list = list(reader)
    scene_classes_dict = {int(item[0]): item[1] for item in scene_classes_list}
    return scene_classes_dict

# 加载分类模型
def load_classification_model(model_weights_path):
    model = build_model()  # 假设 build_model 是从你的模型文件中导入的
    model.load_weights(model_weights_path)
    return model

# 使用 ONNX Runtime 进行场景分割推断
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

    # 后处理：将原始输出转换为分割输出
    seg_map = np.argmax(raw_prediction, axis=3)  # 在类别维度上取 argmax
    return np.squeeze(seg_map).astype(np.int8)  # 移除多余维度并转换类型

# 创建 Cityscapes 标签色图
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

# 更新显示图像
def update_image_display(image_path, image_label):
    image = Image.open(image_path)
    image = image.resize((512, 256))  # 调整图像大小以适应窗口
    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk

# 选择文件
def open_file(image_label):
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        print(f"Selected file: {file_path}")
        update_image_display(file_path, image_label)
        return file_path
    return None

# 进行场景分割
def scene_segmentation(image_path, image_label1, image_label2,  model_path):
    image = Image.open(image_path)
    seg_map = predict(image, model_path)  # 预测分割图

    # 原图
    update_image_display(image_path, image_label1)

    # 分割图
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = Image.fromarray(seg_image)
    seg_image = seg_image.resize((512, 256))
    seg_image_tk = ImageTk.PhotoImage(seg_image)
    image_label2.config(image=seg_image_tk)
    image_label2.image = seg_image_tk

# 选择图片并进行分类
def classify_image(model, scene_classes_dict, image_path, result_label):
    image = Image.open(image_path)
    rgb_img = np.asarray(image.convert('RGB'))
    rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
    rgb_img = preprocess_input(rgb_img)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    class_id = np.argmax(preds)
    class_name = scene_classes_dict.get(class_id, "Unknown")

    # 更新分类结果显示
    result_text = f"Predicted Class: {class_name}"
    result_label.config(text=result_text)

# 主界面
def create_gui():
    window = tk.Tk()
    window.title("场景分类与分割")

    # 设置窗口大小
    window.geometry("1200x600")

    # 加载分类信息
    scene_classes_dict = load_scene_classes()

    # 加载分类模型
    classification_model = load_classification_model(os.path.join('models', 'model.11-0.6262.hdf5'))

    # 加载场景分割模型路径
    segmentation_model_path = 'frozen_inference_graph3.onnx'  # 场景分割模型路径

    # 初始化图像路径
    image_path = None

    # 创建顶部按钮框
    button_frame = tk.Frame(window)
    button_frame.pack(pady=20)

    # 创建“选择图片”按钮
    def choose_image():
        nonlocal image_path
        image_path = open_file(image_label1)  # 选择图片并更新路径

    select_button = tk.Button(button_frame, text="加载图片", command=choose_image, height=2, width=40)
    select_button.pack(side=tk.LEFT, padx=10)

    # 创建“场景分割”按钮
    def segmentation():
        if image_path:
            scene_segmentation(image_path, image_label1, image_label2, segmentation_model_path)  # 调用场景分割
        else:
            messagebox.showwarning("No Image", "Please select an image first.")

    segmentation_button = tk.Button(button_frame, text="场景分割", command=segmentation, height=2, width=40)
    segmentation_button.pack(side=tk.LEFT, padx=10)

    # 创建“场景分类”按钮
    def classify():
        if image_path:
            classify_image(classification_model, scene_classes_dict, image_path, result_label)
        else:
            messagebox.showwarning("No Image", "Please select an image first.")

    classify_button = tk.Button(button_frame, text="场景分类", command=classify, height=2, width=40)
    classify_button.pack(side=tk.LEFT, padx=10)

    # 创建图像显示区域
    image_frame = tk.Frame(window)
    image_frame.pack(pady=20, fill='y', expand=True)

    # 显示原图的标签
    image_label1 = Label(image_frame)
    image_label1.grid(row=0, column=0, padx=10)

    # 显示分割图的标签
    image_label2 = Label(image_frame)
    image_label2.grid(row=0, column=1, padx=10)

    # 创建标签显示分类结果
    result_label = tk.Label(window, text="分类结果将在这里显示", font=("Arial", 12))
    result_label.pack(pady=10)

    # 使用空白框来占据剩余空间，确保内容垂直居中
    spacer = tk.Label(window, text="", height=1)
    spacer.pack(expand=True)

    # 运行主界面
    window.mainloop()

# 运行程序
if __name__ == "__main__":
    create_gui()
