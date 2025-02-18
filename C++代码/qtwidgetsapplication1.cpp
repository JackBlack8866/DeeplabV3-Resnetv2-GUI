#include "qtwidgetsapplication1.h"

QtWidgetsApplication1::QtWidgetsApplication1(QWidget *parent)
	: QWidget(parent)
{
	// ���������֣���ֱ���� (VLayout) ���ڰ�ť��ˮƽ���� (HLayout) �������ҷָ�ԭͼ��ָ���ͼ��
	QVBoxLayout* mainLayout = new QVBoxLayout(this); // ������
	QHBoxLayout* imageLayout = new QHBoxLayout(); // ����������ʾͼƬ�Ĳ���

	// ������ť���� (ˮƽ���֣�����������ť)
	QHBoxLayout* buttonLayout = new QHBoxLayout();
	selectImageButton = new QPushButton(QString::fromUtf8("ѡ��ͼƬ"), this);
	segmentButton = new QPushButton(QString::fromUtf8("���зָ�"), this);

	// ���ð�ť�ĸ߶�
	selectImageButton->setFixedHeight(200);  // ���ø߶�Ϊ200
	segmentButton->setFixedHeight(200);       // ���ø߶�Ϊ200

	buttonLayout->addWidget(selectImageButton);
	buttonLayout->addWidget(segmentButton);
	mainLayout->addLayout(buttonLayout);

	// ������ʾԭͼ�� QLabel
	originalImageLabel = new QLabel(this);
	originalImageLabel->setAlignment(Qt::AlignCenter);
	originalImageLabel->setText(QString::fromUtf8("ԭͼ����������ʾ"));
	imageLayout->addWidget(originalImageLabel);

	// ������ʾ�ָ�ͼ��� QLabel
	segmentedImageLabel = new QLabel(this);
	segmentedImageLabel->setAlignment(Qt::AlignCenter);
	segmentedImageLabel->setText(QString::fromUtf8("�ָ�������������ʾ"));
	imageLayout->addWidget(segmentedImageLabel);

	// ��ͼƬ��ʾ������ӵ�������
	mainLayout->addLayout(imageLayout);

	// ���Ӱ�ť����źŵ��ۺ���
	connect(selectImageButton, &QPushButton::clicked, this, &QtWidgetsApplication1::onSelectImage);
	connect(segmentButton, &QPushButton::clicked, this, &QtWidgetsApplication1::onRunSegmentation);
}

QtWidgetsApplication1::~QtWidgetsApplication1()
{
	// ������Դ������У�
}

void QtWidgetsApplication1::onSelectImage()
{
	// ���ļ��Ի���ѡ��ͼƬ
	QString imagePath = QFileDialog::getOpenFileName(this, QString::fromUtf8("ѡ��ͼƬ"), "", "Images (*.png *.jpg *.jpeg)");
	if (imagePath.isEmpty()) {
		originalImageLabel->setText(QString::fromUtf8("δѡ��ͼƬ��"));
		return;
	}

	// ����ѡ����ͼƬ·������ʾ��ԭͼ QLabel ��
	selectedImagePath = imagePath;
	QImage image(selectedImagePath);
	originalImageLabel->setPixmap(QPixmap::fromImage(image).scaled(originalImageLabel->size(), Qt::KeepAspectRatio));
	originalImageLabel->setText(""); // ���Ĭ���ı�
}

void QtWidgetsApplication1::onRunSegmentation()
{
	if (selectedImagePath.isEmpty()) {
		segmentedImageLabel->setText(QString::fromUtf8("����ѡ��ͼƬ��"));
		return;
	}

	// ���� deeplabV3 ����
	deeplabV3 deeplab;

	// ���� runSegmentation ����
	cv::Mat segImage = deeplab.runSegmentation(selectedImagePath.toStdString());
	if (segImage.empty()) {
		segmentedImageLabel->setText(QString::fromUtf8("�ָ�ʧ�ܣ�����ģ�ͺ�ͼƬ·����"));
		return;
	}

	// ���ָ���ת��Ϊ QImage
	cv::cvtColor(segImage, segImage, cv::COLOR_BGR2RGB); // ת��Ϊ RGB ��ʽ
	QImage qImage((const uchar*)segImage.data, segImage.cols, segImage.rows, segImage.step, QImage::Format_RGB888);

	// �� QLabel ����ʾ�ָ���
	segmentedImageLabel->setPixmap(QPixmap::fromImage(qImage).scaled(segmentedImageLabel->size(), Qt::KeepAspectRatio));
	segmentedImageLabel->setText(""); // ���Ĭ���ı�
}
