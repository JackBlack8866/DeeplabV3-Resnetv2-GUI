#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "segment.h"

using namespace cv;
using namespace std;
using namespace Ort;


// ���� Cityscapes ����ɫӳ��
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


// ʹ�� ONNX Runtime �����ƶ�
std::vector<int> deeplabV3::predict(const cv::Mat& image) {

	//����ģ�͵�·��
	const std::string model_path = "frozen_inference_graph3.onnx";

	// ���� ONNX Runtime ����
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

	// ��ģ��·��ת��Ϊ wstring ����
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	const wchar_t* model_path_w = widestr.c_str();

	// ���� ONNX Runtime �Ự
	Ort::SessionOptions session_options;
	Ort::Session session(env, model_path_w, session_options);

	// ����ͼƬ��С��Ԥ����
	cv::Mat resized_image;
	cv::resize(image, resized_image, cv::Size(2049, 1025));  // ����Ϊģ�������С
	resized_image.convertTo(resized_image, CV_32F, 1.0 / 127.5, -1);  // ��һ������

	// ׼����������
	std::vector<float> input_tensor_values(resized_image.total() * resized_image.channels());
	std::memcpy(input_tensor_values.data(), resized_image.data, input_tensor_values.size() * sizeof(float));

	std::vector<int64_t> input_shape = { 1, resized_image.rows, resized_image.cols, resized_image.channels() };  // [batch_size, height, width, channels]

	// ������������
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

	// ��ȡ������������
	Ort::AllocatorWithDefaultOptions allocator;
	const char* input_name = session.GetInputName(0, allocator);
	const char* output_name = session.GetOutputName(0, allocator);

	// �����ƶ�
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, &output_name, 1);

	// ��ȡ���
	float* raw_prediction = output_tensors.front().GetTensorMutableData<float>();

	// ��ӡԭʼԤ����״
	std::cout << "Raw prediction shape: 1, 1025, 2049, 19" << std::endl; // ����ģ���������

// ʹ�� cv::Mat �ͱ�׼�⺯��ʵ�� np.argmax �� np.squeeze
	cv::Mat prediction_mat(1025, 2049 * 19, CV_32FC1, raw_prediction);
	cv::Mat max_indices(1025, 2049, CV_32SC1); // �������ڴ�����ֵ�����ľ���

	// ��ÿ��λ�û�ȡ���ֵ����Ӧ������
	for (int i = 0; i < 1025; ++i) {
		for (int j = 0; j < 2049; ++j) {
			int max_index = 0; // ��¼���ֵ������
			float max_value = raw_prediction[i * 2049 * 19 + j * 19]; // ��ʼ���ֵ

			// ����ÿ�����ص�19��Ԥ��ֵ�е����ֵ������
			for (int k = 1; k < 19; ++k) {
				float value = raw_prediction[i * 2049 * 19 + j * 19 + k];
				if (value > max_value) {
					max_value = value;
					max_index = k; // �������ֵ������
				}
			}

			max_indices.at<int>(i, j) = max_index; // �洢���ֵ����
		}
	}

	cv::Mat seg_image = max_indices; // ����������ֱ����Ϊ�ָ�ͼ����

	return std::vector<int>(seg_image.begin<int>(), seg_image.end<int>()); // ���طָ�ͼ
}

// ���ָ���ת��Ϊ��ɫͼ��
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
	// ���ز���������ͼƬ
	cv::Mat image = cv::imread(IMAGE_PATH);
	if (image.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
		return cv::Mat();  // ���ؿ�ͼ��
	}

	// �����ƶ�
	std::vector<int> seg_map = predict(image);

	// ���ӻ��ָ���
	cv::Mat seg_image = visualizeSegmentation(seg_map, 1025, 2049);
	cv::resize(seg_image, seg_image, image.size());

	return seg_image;  // ���طָ���ͼ��
}


// ����ģ�ͺ�����ͼƬ��·��
//const std::string MODEL_PATH = "frozen_inference_graph3.onnx";  // �滻Ϊ����ģ��·��
//const std::string IMAGE_PATH = "frankfurt_000000_000294_leftImg8bit.png";  // �滻Ϊ��������ͼƬ·��
//
////int main() { 
////    // ���ز���������ͼƬ
////    cv::Mat image = cv::imread(IMAGE_PATH);
////    if (image.empty()) {
////        std::cerr << "Could not open or find the image!" << std::endl;
////        return -1;
////    }
////
////    // ���ͼ��Ĵ�С
////    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
////
////    // �����ƶ�
////    std::vector<int> seg_map = predict(image, MODEL_PATH);
////
////    // ���ӻ��ָ���
////    cv::Mat seg_image = visualizeSegmentation(seg_map, 1025, 2049);
////	cv::resize(seg_image, seg_image, image.size());
////
////     //��ʾ���
////    cv::imshow("Input Image", image);
////    cv::imshow("Segmentation Map", seg_image);
////	// ����ָ���ͼ��
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
