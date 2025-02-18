#ifndef SEGMENT_H
#define SEGMENT_H

#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


#ifdef IMAQDLL_EXPORTS
#define IMAQDLL	__declspec(dllexport)
#else
#define IMAQDLL __declspec(dllimport)
#endif

using namespace cv;
using namespace std;
using namespace Ort;

class IMAQDLL deeplabV3 {
public:
	// 创建 Cityscapes 颜色映射
	cv::Mat createCityscapesLabelColormap();

	// 使用 ONNX Runtime 进行图像分割
	std::vector<int> predict(const cv::Mat& image);

	// 将分割结果转换为彩色图像
	cv::Mat visualizeSegmentation(const std::vector<int>& seg_map, int height, int width);

	// 封装函数进行分割处理，并返回分割结果图像
	cv::Mat runSegmentation( const std::string& IMAGE_PATH);

private:
	Ort::Env env;                  // ONNX Runtime 环境
	Ort::Session* session;         // ONNX Runtime 会话
	Ort::SessionOptions session_options; // ONNX Runtime 会话选项
	std::vector<const char*> input_node_names;  // 输入节点名称
	std::vector<const char*> output_node_names; // 输出节点名称
	int input_width;               // 模型要求的输入宽度
	int input_height;              // 模型要求的输入高度
	int num_classes;               // 分类数量

};
#endif // SEGMENT_H





