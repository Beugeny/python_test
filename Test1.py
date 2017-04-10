import sys
from PyQt5 import Qt

from PyQt5 import QtCore
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QAbstractTableModel
from PyQt5.QtCore import QDir
from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtWidgets import QMainWindow
from spyder.widgets.variableexplorer.dataframeeditor import DataFrameModel

import qdarkstyle
from ui.TestUI import Ui_MainWindow


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(TableModel, self).__init__()
        self.datatable = None

    def update(self, dataIn):
        self.datatable = dataIn

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.columns.values)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            i = index.row()
            j = index.column()
            return '{0}'.format(self.datatable.iloc[i, j])
        elif role == QtCore.Qt.ToolTipRole:
            i = index.row()
            j = index.column()
            return 'Index={0}-{1}'.format(i, j)
        else:
            return QtCore.QVariant()

    def flags(self, index):
        if index.isValid():
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled

    def setData(self, index, value, role=None):
        if index.isValid() and role == QtCore.Qt.EditRole:
            i = index.row()
            j = index.column()
            self.datatable.iloc[i, j] = value
            return True
        return False

    def headerData(self, p_int, Qt_Orientation, role=None):
        if role == QtCore.Qt.DisplayRole:
            if Qt_Orientation == QtCore.Qt.Horizontal:
                return self.datatable.columns.values[p_int]
            else:
                return self.datatable.index[p_int]
        return QtCore.QVariant()


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        w = Ui_MainWindow()
        w.setupUi(self)

        m = TableModel()
        m.update(pd.DataFrame({"A": [10, 20, 30, 40], "B": ["one", "two", "three", "four"]}))
        # m = DataFrameModel(pd.DataFrame({"A": [10, 20, 30, 40], "B": ["one", "two", "three", "four"]}))
        # m.update(pd.DataFrame({"A": [10, 20, 30, 40], "B": ["one", "two", "three", "four"]}))

        print(m.data(m.index(0, 1), QtCore.Qt.DecorationRole))

        w.tableView.setModel(m)
        w.tableView_2.setModel(m)
        print(w.tableView.itemDelegate())
        w.tableView_2.setSelectionModel(w.tableView.selectionModel())

        model = QFileSystemModel()

        model.setRootPath(QDir.currentPath())
        w.treeView.setModel(model)


app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
w = MyWindow()
w.show()

sys.exit(app.exec_())
