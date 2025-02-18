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
	void onSelectImage(); // 选择图片按钮的槽函数
	void onRunSegmentation(); // 运行分割按钮的槽函数

private:
	QLabel* originalImageLabel; // 显示原图
	QLabel* segmentedImageLabel; // 显示分割后的图像
	QPushButton* selectImageButton; // 选择图片按钮
	QPushButton* segmentButton; // 运行分割按钮
	QString selectedImagePath; // 保存选定的图片路径
};
