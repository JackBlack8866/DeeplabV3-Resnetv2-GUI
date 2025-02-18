#pragma once

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QString>
#include "segment.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1600)
# pragma execution_character_set("utf-8")
#endif


class QtWidgetsApplication1 : public QWidget
{
	Q_OBJECT

public:
	QtWidgetsApplication1(QWidget *parent = nullptr);
	~QtWidgetsApplication1();

private slots:
	void onSelectImage(); // ѡ��ͼƬ��ť�Ĳۺ���
	void onRunSegmentation(); // ���зָť�Ĳۺ���

private:
	QLabel* originalImageLabel; // ��ʾԭͼ
	QLabel* segmentedImageLabel; // ��ʾ�ָ���ͼ��
	QPushButton* selectImageButton; // ѡ��ͼƬ��ť
	QPushButton* segmentButton; // ���зָť
	QString selectedImagePath; // ����ѡ����ͼƬ·��
};
