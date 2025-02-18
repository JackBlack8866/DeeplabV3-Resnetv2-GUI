#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "segment.h"

using namespace cv;
using namespace std;
using namespace Ort;


// 创建 Cityscapes 的颜色映射
cv::Mat deeplabV3::createCityscapesLabelColormap() {
	cv::Mat colormap(19, 1, CV_8UC3);
	colormap.at<cv::Vec3b>(0) = cv::Vec3b(128, 64, 128);   // road
	colormap.at<cv::Vec3b>(1) = cv::Vec3b(232, 35, 244);   // sidewalk
	colormap.at<cv::Vec3b>(2) = cv::Vec3b(70, 70, 70);     // building
	colormap.at<cv::Vec3b>(3) = cv::Vec3b(156, 102, 102);  // wall
	colormap.at<cv::Vec3b>(4) = cv::Vec3b(153, 153, 190);  // fence
	colormap.at<cv::Vec3b>(5) = cv::Vec3b(153, 153, 153);  // pole
	colormap.at<cv::Vec3b>(6) = cv::Vec3b(30, 170, 250);   // traffic light
	colormap.at<cv::Vec3b>(7) = cv::Vec3b(0, 220, 220);    // traffic sign
	colormap.at<cv::Vec3b>(8) = cv::Vec3b(35, 142, 107);   // vegetation
	colormap.at<cv::Vec3b>(9) = cv::Vec3b(152, 251, 152);  // terrain
	colormap.at<cv::Vec3b>(10) = cv::Vec3b(180, 130, 70);  // sky
	colormap.at<cv::Vec3b>(11) = cv::Vec3b(60, 20, 220);   // person
	colormap.at<cv::Vec3b>(12) = cv::Vec3b(0, 0, 255);     // rider
	colormap.at<cv::Vec3b>(13) = cv::Vec3b(142, 0, 0);     // car
	colormap.at<cv::Vec3b>(14) = cv::Vec3b(70, 0, 0);      // truck
	colormap.at<cv::Vec3b>(15) = cv::Vec3b(100, 60, 0);    // bus
	colormap.at<cv::Vec3b>(16) = cv::Vec3b(100, 80, 0);    // train
	colormap.at<cv::Vec3b>(17) = cv::Vec3b(230, 0, 0);     // motorcycle
	colormap.at<cv::Vec3b>(18) = cv::Vec3b(32, 11, 119);   // bicycle
	return colormap;
}


// 使用 ONNX Runtime 进行推断
std::vector<int> deeplabV3::predict(const cv::Mat& image) {

	//设置模型的路径
	const std::string model_path = "frozen_inference_graph3.onnx";

	// 创建 ONNX Runtime 环境
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

	// 将模型路径转换为 wstring 类型
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	const wchar_t* model_path_w = widestr.c_str();

	// 创建 ONNX Runtime 会话
	Ort::SessionOptions session_options;
	Ort::Session session(env, model_path_w, session_options);

	// 调整图片大小并预处理
	cv::Mat resized_image;
	cv::resize(image, resized_image, cv::Size(2049, 1025));  // 调整为模型输入大小
	resized_image.convertTo(resized_image, CV_32F, 1.0 / 127.5, -1);  // 归一化处理

	// 准备输入数据
	std::vector<float> input_tensor_values(resized_image.total() * resized_image.channels());
	std::memcpy(input_tensor_values.data(), resized_image.data, input_tensor_values.size() * sizeof(float));

	std::vector<int64_t> input_shape = { 1, resized_image.rows, resized_image.cols, resized_image.channels() };  // [batch_size, height, width, channels]

	// 创建输入张量
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

	// 获取输入和输出名称
	Ort::AllocatorWithDefaultOptions allocator;
	const char* input_name = session.GetInputName(0, allocator);
	const char* output_name = session.GetOutputName(0, allocator);

	// 进行推断
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, &output_name, 1);

	// 获取输出
	float* raw_prediction = output_tensors.front().GetTensorMutableData<float>();

	// 打印原始预测形状
	std::cout << "Raw prediction shape: 1, 1025, 2049, 19" << std::endl; // 根据模型输出调整

// 使用 cv::Mat 和标准库函数实现 np.argmax 和 np.squeeze
	cv::Mat prediction_mat(1025, 2049 * 19, CV_32FC1, raw_prediction);
	cv::Mat max_indices(1025, 2049, CV_32SC1); // 创建用于存放最大值索引的矩阵

	// 对每个位置获取最大值和相应的索引
	for (int i = 0; i < 1025; ++i) {
		for (int j = 0; j < 2049; ++j) {
			int max_index = 0; // 记录最大值的索引
			float max_value = raw_prediction[i * 2049 * 19 + j * 19]; // 初始最大值

			// 查找每个像素的19个预测值中的最大值和索引
			for (int k = 1; k < 19; ++k) {
				float value = raw_prediction[i * 2049 * 19 + j * 19 + k];
				if (value > max_value) {
					max_value = value;
					max_index = k; // 更新最大值的索引
				}
			}

			max_indices.at<int>(i, j) = max_index; // 存储最大值索引
		}
	}

	cv::Mat seg_image = max_indices; // 将索引矩阵直接作为分割图返回

	return std::vector<int>(seg_image.begin<int>(), seg_image.end<int>()); // 返回分割图
}

// 将分割结果转换为彩色图像
cv::Mat deeplabV3::visualizeSegmentation(const std::vector<int>& seg_map, int height, int width) {
	cv::Mat color_image(height, width, CV_8UC3);
	cv::Mat colormap = createCityscapesLabelColormap();

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int label = seg_map[i * width + j];
			color_image.at<cv::Vec3b>(i, j) = colormap.at<cv::Vec3b>(label);
		}
	}

	return color_image;
}

cv::Mat deeplabV3::runSegmentation(const std::string& IMAGE_PATH) {
	// 加载并处理输入图片
	cv::Mat image = cv::imread(IMAGE_PATH);
	if (image.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
		return cv::Mat();  // 返回空图像
	}

	// 进行推断
	std::vector<int> seg_map = predict(image);

	// 可视化分割结果
	cv::Mat seg_image = visualizeSegmentation(seg_map, 1025, 2049);
	cv::resize(seg_image, seg_image, image.size());

	return seg_image;  // 返回分割结果图像
}


// 设置模型和输入图片的路径
//const std::string MODEL_PATH = "frozen_inference_graph3.onnx";  // 替换为您的模型路径
//const std::string IMAGE_PATH = "frankfurt_000000_000294_leftImg8bit.png";  // 替换为您的输入图片路径
//
////int main() { 
////    // 加载并处理输入图片
////    cv::Mat image = cv::imread(IMAGE_PATH);
////    if (image.empty()) {
////        std::cerr << "Could not open or find the image!" << std::endl;
////        return -1;
////    }
////
////    // 输出图像的大小
////    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
////
////    // 进行推断
////    std::vector<int> seg_map = predict(image, MODEL_PATH);
////
////    // 可视化分割结果
////    cv::Mat seg_image = visualizeSegmentation(seg_map, 1025, 2049);
////	cv::resize(seg_image, seg_image, image.size());
////
////     //显示结果
////    cv::imshow("Input Image", image);
////    cv::imshow("Segmentation Map", seg_image);
////	// 保存分割结果图像
////	//cv::imwrite("segmentation_input.png", image);
////	//cv::imwrite("segmentation_map.png", seg_image);
////	std::cout << "Segmentation map saved as segmentation_map.png" << std::endl;
////
////    cv::waitKey(0);
////
////    return 0;
////}
//
//
//
//
//
//
//
