#include "qtwidgetsapplication1.h"
#include "segment.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	QtWidgetsApplication1 w; // 创建主窗口
	w.show();                // 显示窗口
	return a.exec();         // 运行 Qt 应用
}
