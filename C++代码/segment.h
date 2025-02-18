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
	// ���� Cityscapes ��ɫӳ��
	cv::Mat createCityscapesLabelColormap();

	// ʹ�� ONNX Runtime ����ͼ��ָ�
	std::vector<int> predict(const cv::Mat& image);

	// ���ָ���ת��Ϊ��ɫͼ��
	cv::Mat visualizeSegmentation(const std::vector<int>& seg_map, int height, int width);

	// ��װ�������зָ�������طָ���ͼ��
	cv::Mat runSegmentation( const std::string& IMAGE_PATH);

private:
	Ort::Env env;                  // ONNX Runtime ����
	Ort::Session* session;         // ONNX Runtime �Ự
	Ort::SessionOptions session_options; // ONNX Runtime �Ựѡ��
	std::vector<const char*> input_node_names;  // ����ڵ�����
	std::vector<const char*> output_node_names; // ����ڵ�����
	int input_width;               // ģ��Ҫ���������
	int input_height;              // ģ��Ҫ�������߶�
	int num_classes;               // ��������

};
#endif // SEGMENT_H





