#include "qtwidgetsapplication1.h"

QtWidgetsApplication1::QtWidgetsApplication1(QWidget *parent)
	: QWidget(parent)
{
	// 设置主布局：垂直布局 (VLayout) 用于按钮，水平布局 (HLayout) 用于左右分隔原图与分割后的图像
	QVBoxLayout* mainLayout = new QVBoxLayout(this); // 主布局
	QHBoxLayout* imageLayout = new QHBoxLayout(); // 用于左右显示图片的布局

	// 创建按钮布局 (水平布局，包含两个按钮)
	QHBoxLayout* buttonLayout = new QHBoxLayout();
	selectImageButton = new QPushButton(QString::fromUtf8("选择图片"), this);
	segmentButton = new QPushButton(QString::fromUtf8("运行分割"), this);

	// 设置按钮的高度
	selectImageButton->setFixedHeight(200);  // 设置高度为200
	segmentButton->setFixedHeight(200);       // 设置高度为200

	buttonLayout->addWidget(selectImageButton);
	buttonLayout->addWidget(segmentButton);
	mainLayout->addLayout(buttonLayout);

	// 创建显示原图的 QLabel
	originalImageLabel = new QLabel(this);
	originalImageLabel->setAlignment(Qt::AlignCenter);
	originalImageLabel->setText(QString::fromUtf8("原图将在这里显示"));
	imageLayout->addWidget(originalImageLabel);

	// 创建显示分割图像的 QLabel
	segmentedImageLabel = new QLabel(this);
	segmentedImageLabel->setAlignment(Qt::AlignCenter);
	segmentedImageLabel->setText(QString::fromUtf8("分割结果将在这里显示"));
	imageLayout->addWidget(segmentedImageLabel);

	// 将图片显示布局添加到主布局
	mainLayout->addLayout(imageLayout);

	// 连接按钮点击信号到槽函数
	connect(selectImageButton, &QPushButton::clicked, this, &QtWidgetsApplication1::onSelectImage);
	connect(segmentButton, &QPushButton::clicked, this, &QtWidgetsApplication1::onRunSegmentation);
}

QtWidgetsApplication1::~QtWidgetsApplication1()
{
	// 清理资源（如果有）
}

void QtWidgetsApplication1::onSelectImage()
{
	// 打开文件对话框选择图片
	QString imagePath = QFileDialog::getOpenFileName(this, QString::fromUtf8("选择图片"), "", "Images (*.png *.jpg *.jpeg)");
	if (imagePath.isEmpty()) {
		originalImageLabel->setText(QString::fromUtf8("未选择图片！"));
		return;
	}

	// 保存选定的图片路径并显示在原图 QLabel 上
	selectedImagePath = imagePath;
	QImage image(selectedImagePath);
	originalImageLabel->setPixmap(QPixmap::fromImage(image).scaled(originalImageLabel->size(), Qt::KeepAspectRatio));
	originalImageLabel->setText(""); // 清除默认文本
}

void QtWidgetsApplication1::onRunSegmentation()
{
	if (selectedImagePath.isEmpty()) {
		segmentedImageLabel->setText(QString::fromUtf8("请先选择图片！"));
		return;
	}

	// 创建 deeplabV3 对象
	deeplabV3 deeplab;

	// 调用 runSegmentation 函数
	cv::Mat segImage = deeplab.runSegmentation(selectedImagePath.toStdString());
	if (segImage.empty()) {
		segmentedImageLabel->setText(QString::fromUtf8("分割失败！请检查模型和图片路径。"));
		return;
	}

	// 将分割结果转换为 QImage
	cv::cvtColor(segImage, segImage, cv::COLOR_BGR2RGB); // 转换为 RGB 格式
	QImage qImage((const uchar*)segImage.data, segImage.cols, segImage.rows, segImage.step, QImage::Format_RGB888);

	// 在 QLabel 中显示分割结果
	segmentedImageLabel->setPixmap(QPixmap::fromImage(qImage).scaled(segmentedImageLabel->size(), Qt::KeepAspectRatio));
	segmentedImageLabel->setText(""); // 清除默认文本
}
