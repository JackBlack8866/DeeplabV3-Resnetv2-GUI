#include "qtwidgetsapplication1.h"
#include "segment.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	QtWidgetsApplication1 w; // ����������
	w.show();                // ��ʾ����
	return a.exec();         // ���� Qt Ӧ��
}
