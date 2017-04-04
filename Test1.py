import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStyleFactory

import qdarkstyle


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def on_btn_click(self):
        self.l.setText("Я нажат, я нажат, очееееень длинный текст")

    def init_ui(self):
        self.b = QtWidgets.QPushButton("Push button")
        self.b.clicked.connect(self.on_btn_click)
        self.l = QtWidgets.QLabel("Label")

        h_box = QtWidgets.QHBoxLayout()
        h_box.addStretch()
        h_box.addWidget(self.b)
        h_box.addStretch()
        h_box.addWidget(self.l)
        h_box.addStretch()

        v_box = QtWidgets.QVBoxLayout()
        v_box.addLayout(h_box)

        self.le = QtWidgets.QLineEdit()
        self.le.textChanged.connect(lambda: self.b.setText(self.sender().text()))

        self.s1 = QtWidgets.QSlider(Qt.Horizontal)
        self.s1.setMinimum(1)
        self.s1.setMaximum(100)
        self.s1.setValue(25)
        self.s1.setTickInterval(10)
        self.s1.setTickPosition(QtWidgets.QSlider.TicksBelow)

        v_box.addWidget(self.s1)
        v_box.addWidget(self.le)

        self.setLayout(v_box)
        self.setWindowTitle('Заголовок окна')


app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
w = MyWindow()
w.show()


sys.exit(app.exec_())
