# import sys
# from qt_test.mainwindow import Ui_MainWindow
# from PyQt5 import QtGui
# from PyQt5.QtWidgets import QMainWindow
#
#
# class WindowExample(QMainWindow, Ui_MainWindow):
#     def __init__(self, parent=None):
#         super(Ui_MainWindow, self).__init__(parent)
#         self.setupUi(self)
#
#
# def main():
#     app = QtGui.QGuiApplication(sys.argv)
#     form = WindowExample()
#     form.show()
#     app.exec_()
#
#
# main()
import sys
from PyQt5.QtWidgets import  QApplication,QDialog,QMainWindow
from qt_test.mainwindow import Ui_MainWindow

app=QApplication(sys.argv)
w=QMainWindow()
w.setWindowTitle("Hello World")

ui=Ui_MainWindow()
ui.setupUi(w)

w.show()

sys.exit(app.exec_())