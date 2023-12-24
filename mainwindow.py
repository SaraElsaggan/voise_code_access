# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1291, 796)
        MainWindow.setStyleSheet("/*Copyright (c) DevSec Studio. All rights reserved.\n"
"\n"
"MIT License\n"
"\n"
"Permission is hereby granted, free of charge, to any person obtaining a copy\n"
"of this software and associated documentation files (the \"Software\"), to deal\n"
"in the Software without restriction, including without limitation the rights\n"
"to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
"copies of the Software, and to permit persons to whom the Software is\n"
"furnished to do so, subject to the following conditions:\n"
"\n"
"The above copyright notice and this permission notice shall be included in all\n"
"copies or substantial portions of the Software.\n"
"\n"
"THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
"IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
"AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
"LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
"OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
"*/\n"
"\n"
"/*-----QWidget-----*/\n"
"QWidget\n"
"{\n"
"    background-color: #fff;\n"
"    color: red;\n"
"    border-radius: 20px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QLabel-----*/\n"
"QLabel\n"
"{\n"
"    background-color: transparent;\n"
"    color: #454544;\n"
"    font-weight: bold;\n"
"    font-size: 20px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QPushButton-----*/\n"
"QPushButton\n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"    font-size: 20px;\n"
"    font-weight: bold;\n"
"    border-top-right-radius: 15px;\n"
"    border-top-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"    border-bottom-left-radius: 15px;\n"
"    padding: 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::disabled\n"
"{\n"
"    background-color: #5c5c5c;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color: #5564f2;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::pressed\n"
"{\n"
"    background-color: #3d4ef2;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QCheckBox-----*/\n"
"QCheckBox\n"
"{\n"
"    background-color: transparent;\n"
"    color: #5c55e9;\n"
"    font-size: 10px;\n"
"    font-weight: bold;\n"
"    border: none;\n"
"    border-radius: 5px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QCheckBox-----*/\n"
"QCheckBox::indicator\n"
"{\n"
"    background-color: #323232;\n"
"    border: 1px solid darkgray;\n"
"    width: 12px;\n"
"    height: 12px;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:checked\n"
"{\n"
"    image:url(\"./ressources/check.png\");\n"
"    background-color: #5c55e9;\n"
"    border: 1px solid #5c55e9;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:unchecked:hover\n"
"{\n"
"    border: 1px solid #5c55e9;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::disabled\n"
"{\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:disabled\n"
"{\n"
"    background-color: #656565;\n"
"    color: #656565;\n"
"    border: 1px solid #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QLineEdit-----*/\n"
"QLineEdit\n"
"{\n"
"    background-color: #c2c7d5;\n"
"    color: #2a547f;\n"
"    border: none;\n"
"    padding: 5px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QListView-----*/\n"
"QListView\n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    show-decoration-selected: 0;\n"
"    border-radius: 4px;\n"
"    padding-left: -15px;\n"
"    padding-right: -15px;\n"
"    padding-top: 5px;\n"
"\n"
"} \n"
"\n"
"\n"
"QListView:disabled \n"
"{\n"
"    background-color: #5c5c5c;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item\n"
"{\n"
"    background-color: #454e5e;\n"
"    border: none;\n"
"    padding: 10px;\n"
"    border-radius: 0px;\n"
"    padding-left : 10px;\n"
"    height: 32px;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item:selected\n"
"{\n"
"    color: #000;\n"
"    background-color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item:!selected\n"
"{\n"
"    color:white;\n"
"    background-color: transparent;\n"
"    border: none;\n"
"    padding-left : 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item:!selected:hover\n"
"{\n"
"    color: #fff;\n"
"    background-color: #5564f2;\n"
"    border: none;\n"
"    padding-left : 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QTreeView-----*/\n"
"QTreeView \n"
"{\n"
"    background-color: #fff;\n"
"    show-decoration-selected: 0;\n"
"    color: #454544;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView:disabled\n"
"{\n"
"    background-color: #242526;\n"
"    show-decoration-selected: 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item \n"
"{\n"
"    border-top-color: transparent;\n"
"    border-bottom-color: transparent;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:hover \n"
"{\n"
"    background-color: #bcbdbb;\n"
"    color: #000;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:selected \n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:selected:active\n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:selected:disabled\n"
"{\n"
"    background-color: #525251;\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::branch:has-children:!has-siblings:closed,\n"
"QTreeView::branch:closed:has-children:has-siblings \n"
"{\n"
"    image: url(://tree-closed.png);\n"
"\n"
"}\n"
"\n"
"QTreeView::branch:open:has-children:!has-siblings,\n"
"QTreeView::branch:open:has-children:has-siblings  \n"
"{\n"
"    image: url(://tree-open.png);\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QTableView & QTableWidget-----*/\n"
"QTableView\n"
"{\n"
"    background-color: #fff;\n"
"    border: 1px solid gray;\n"
"    color: #454544;\n"
"    gridline-color: gray;\n"
"    outline : 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::disabled\n"
"{\n"
"    background-color: #242526;\n"
"    border: 1px solid #32414B;\n"
"    color: #656565;\n"
"    gridline-color: #656565;\n"
"    outline : 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::item:hover \n"
"{\n"
"    background-color: #bcbdbb;\n"
"    color: #000;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::item:selected \n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::item:selected:disabled\n"
"{\n"
"    background-color: #1a1b1c;\n"
"    border: 2px solid #525251;\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableCornerButton::section\n"
"{\n"
"    background-color: #ced5e3;\n"
"    border: none;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section\n"
"{\n"
"    color: #2a547f;\n"
"    border: 0px;\n"
"    background-color: #ced5e3;\n"
"    padding: 5px;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section:disabled\n"
"{\n"
"    background-color: #525251;\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section:checked\n"
"{\n"
"    color: #fff;\n"
"    background-color: #5c55e9;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section:checked:disabled\n"
"{\n"
"    color: #656565;\n"
"    background-color: #525251;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::vertical::first,\n"
"QHeaderView::section::vertical::only-one\n"
"{\n"
"    border-top: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::vertical\n"
"{\n"
"    border-top: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::horizontal::first,\n"
"QHeaderView::section::horizontal::only-one\n"
"{\n"
"    border-left: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::horizontal\n"
"{\n"
"    border-left: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"QTableView, QTableCornerButton::section,\n"
"QHeaderView::section\n"
"{\n"
"    font: bold 20px;\n"
"}\n"
"\n"
"\n"
"\n"
"/*-----QScrollBar-----*/\n"
"QScrollBar:horizontal \n"
"{\n"
"    background-color: transparent;\n"
"    height: 8px;\n"
"    margin: 0px;\n"
"    padding: 0px;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::handle:horizontal \n"
"{\n"
"    border: none;\n"
"    min-width: 100px;\n"
"    background-color: #7e92b7;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-line:horizontal, \n"
"QScrollBar::sub-line:horizontal,\n"
"QScrollBar::add-page:horizontal, \n"
"QScrollBar::sub-page:horizontal \n"
"{\n"
"    width: 0px;\n"
"    background-color: #d8dce6;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar:vertical \n"
"{\n"
"    background-color: transparent;\n"
"    width: 8px;\n"
"    margin: 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::handle:vertical \n"
"{\n"
"    border: none;\n"
"    min-height: 100px;\n"
"    background-color: #7e92b7;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-line:vertical, \n"
"QScrollBar::sub-line:vertical,\n"
"QScrollBar::add-page:vertical, \n"
"QScrollBar::sub-page:vertical \n"
"{\n"
"    height: 0px;\n"
"    background-color: #d8dce6;\n"
"\n"
"}\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setStyleSheet("border-radius: 20px; /* Adjust the value to control the roundness of the corners */\n"
"    background-color: rgb(226, 241, 255); /* Set the background color */")
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem, 2, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem1, 8, 1, 1, 1)
        self.lbl_access = QtWidgets.QLabel(self.widget_2)
        self.lbl_access.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_access.setObjectName("lbl_access")
        self.gridLayout_4.addWidget(self.lbl_access, 7, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 3, 2, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setHorizontalSpacing(100)
        self.gridLayout.setVerticalSpacing(20)
        self.gridLayout.setObjectName("gridLayout")
        self.lbl_acess_user_6 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_6.setObjectName("lbl_acess_user_6")
        self.gridLayout.addWidget(self.lbl_acess_user_6, 5, 0, 1, 1)
        self.chkBox_user_7 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_7.setText("")
        self.chkBox_user_7.setObjectName("chkBox_user_7")
        self.gridLayout.addWidget(self.chkBox_user_7, 6, 1, 1, 1)
        self.chkBox_user_3 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_3.setText("")
        self.chkBox_user_3.setObjectName("chkBox_user_3")
        self.gridLayout.addWidget(self.chkBox_user_3, 2, 1, 1, 1)
        self.lbl_acess_user_7 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_7.setObjectName("lbl_acess_user_7")
        self.gridLayout.addWidget(self.lbl_acess_user_7, 6, 0, 1, 1)
        self.lbl_acess_user_3 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_3.setObjectName("lbl_acess_user_3")
        self.gridLayout.addWidget(self.lbl_acess_user_3, 2, 0, 1, 1)
        self.lbl_acess_user_5 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_5.setObjectName("lbl_acess_user_5")
        self.gridLayout.addWidget(self.lbl_acess_user_5, 4, 0, 1, 1)
        self.lbl_acess_user_4 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_4.setObjectName("lbl_acess_user_4")
        self.gridLayout.addWidget(self.lbl_acess_user_4, 3, 0, 1, 1)
        self.lbl_acess_user_1 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_1.setStyleSheet("")
        self.lbl_acess_user_1.setObjectName("lbl_acess_user_1")
        self.gridLayout.addWidget(self.lbl_acess_user_1, 0, 0, 1, 1)
        self.chkBox_user_5 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_5.setText("")
        self.chkBox_user_5.setObjectName("chkBox_user_5")
        self.gridLayout.addWidget(self.chkBox_user_5, 4, 1, 1, 1)
        self.chkBox_user_4 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_4.setText("")
        self.chkBox_user_4.setObjectName("chkBox_user_4")
        self.gridLayout.addWidget(self.chkBox_user_4, 3, 1, 1, 1)
        self.chkBox_user_1 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_1.setText("")
        self.chkBox_user_1.setObjectName("chkBox_user_1")
        self.gridLayout.addWidget(self.chkBox_user_1, 0, 1, 1, 1)
        self.lbl_acess_user_2 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_2.setObjectName("lbl_acess_user_2")
        self.gridLayout.addWidget(self.lbl_acess_user_2, 1, 0, 1, 1)
        self.chkBox_user_2 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_2.setText("")
        self.chkBox_user_2.setObjectName("chkBox_user_2")
        self.gridLayout.addWidget(self.chkBox_user_2, 1, 1, 1, 1)
        self.lbl_acess_user_8 = QtWidgets.QLabel(self.widget_2)
        self.lbl_acess_user_8.setObjectName("lbl_acess_user_8")
        self.gridLayout.addWidget(self.lbl_acess_user_8, 7, 0, 1, 1)
        self.chkBox_user_8 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_8.setText("")
        self.chkBox_user_8.setObjectName("chkBox_user_8")
        self.gridLayout.addWidget(self.chkBox_user_8, 7, 1, 1, 1)
        self.chkBox_user_6 = QtWidgets.QCheckBox(self.widget_2)
        self.chkBox_user_6.setText("")
        self.chkBox_user_6.setObjectName("chkBox_user_6")
        self.gridLayout.addWidget(self.chkBox_user_6, 5, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 3, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem3, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget_2)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 1, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem4, 0, 1, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem5, 4, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem6, 6, 1, 1, 1)
        self.btn_record = QtWidgets.QPushButton(self.widget_2)
        self.btn_record.setStyleSheet("background-color: rgb(61, 78, 242);")
        self.btn_record.setObjectName("btn_record")
        self.gridLayout_4.addWidget(self.btn_record, 5, 1, 1, 1)
        self.gridLayout_9.addWidget(self.widget_2, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem7)
        self.layout_spectogrm = QtWidgets.QVBoxLayout()
        self.layout_spectogrm.setObjectName("layout_spectogrm")
        self.horizontalLayout_5.addLayout(self.layout_spectogrm)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem8)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem9 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem9)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setStyleSheet("QLabel {\n"
"    border: 2px solid  rgb(61, 78, 242);\n"
"    border-radius: 4px;\n"
"    padding: 0px;\n"
"    \n"
"}")
        self.widget.setObjectName("widget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lbl_sentence_1 = QtWidgets.QLabel(self.widget)
        self.lbl_sentence_1.setObjectName("lbl_sentence_1")
        self.gridLayout_3.addWidget(self.lbl_sentence_1, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)
        self.lbl_prop_sentence_2 = QtWidgets.QLabel(self.widget)
        self.lbl_prop_sentence_2.setText("")
        self.lbl_prop_sentence_2.setObjectName("lbl_prop_sentence_2")
        self.gridLayout_3.addWidget(self.lbl_prop_sentence_2, 2, 2, 1, 1)
        self.lbl_prop_sentence_1 = QtWidgets.QLabel(self.widget)
        self.lbl_prop_sentence_1.setText("")
        self.lbl_prop_sentence_1.setObjectName("lbl_prop_sentence_1")
        self.gridLayout_3.addWidget(self.lbl_prop_sentence_1, 1, 2, 1, 1)
        self.lbl_sentence_3 = QtWidgets.QLabel(self.widget)
        self.lbl_sentence_3.setObjectName("lbl_sentence_3")
        self.gridLayout_3.addWidget(self.lbl_sentence_3, 3, 0, 1, 1)
        self.lbl_prop_sentence_3 = QtWidgets.QLabel(self.widget)
        self.lbl_prop_sentence_3.setText("")
        self.lbl_prop_sentence_3.setObjectName("lbl_prop_sentence_3")
        self.gridLayout_3.addWidget(self.lbl_prop_sentence_3, 3, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 2, 1, 1)
        self.lbl_sentence_2 = QtWidgets.QLabel(self.widget)
        self.lbl_sentence_2.setObjectName("lbl_sentence_2")
        self.gridLayout_3.addWidget(self.lbl_sentence_2, 2, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem10, 1, 1, 1, 1)
        self.horizontalLayout_4.addWidget(self.widget)
        spacerItem11 = QtWidgets.QSpacerItem(101, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem11)
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setStyleSheet("QLabel {\n"
"    border: 2px solid  rgb(61, 78, 242);\n"
"    border-radius: 4px;\n"
"    padding: 0px;\n"
"    \n"
"}")
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lbl_prob_user_2 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_2.setText("")
        self.lbl_prob_user_2.setObjectName("lbl_prob_user_2")
        self.gridLayout_6.addWidget(self.lbl_prob_user_2, 2, 2, 1, 1)
        self.lbl_prob_user_8 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_8.setText("")
        self.lbl_prob_user_8.setObjectName("lbl_prob_user_8")
        self.gridLayout_6.addWidget(self.lbl_prob_user_8, 8, 2, 1, 1)
        self.lbl_user_24 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_24.setObjectName("lbl_user_24")
        self.gridLayout_6.addWidget(self.lbl_user_24, 1, 0, 1, 1)
        self.lbl_user_21 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_21.setObjectName("lbl_user_21")
        self.gridLayout_6.addWidget(self.lbl_user_21, 2, 0, 1, 1)
        self.lbl_user_18 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_18.setObjectName("lbl_user_18")
        self.gridLayout_6.addWidget(self.lbl_user_18, 7, 0, 1, 1)
        self.lbl_prob_user_3 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_3.setText("")
        self.lbl_prob_user_3.setObjectName("lbl_prob_user_3")
        self.gridLayout_6.addWidget(self.lbl_prob_user_3, 3, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget_3)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 0, 2, 1, 1)
        self.lbl_prob_user_4 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_4.setText("")
        self.lbl_prob_user_4.setObjectName("lbl_prob_user_4")
        self.gridLayout_6.addWidget(self.lbl_prob_user_4, 4, 2, 1, 1)
        self.lbl_prob_user_7 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_7.setText("")
        self.lbl_prob_user_7.setObjectName("lbl_prob_user_7")
        self.gridLayout_6.addWidget(self.lbl_prob_user_7, 7, 2, 1, 1)
        self.lbl_user_19 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_19.setObjectName("lbl_user_19")
        self.gridLayout_6.addWidget(self.lbl_user_19, 4, 0, 1, 1)
        self.lbl_user_20 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_20.setObjectName("lbl_user_20")
        self.gridLayout_6.addWidget(self.lbl_user_20, 6, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        self.label_9.setObjectName("label_9")
        self.gridLayout_6.addWidget(self.label_9, 0, 0, 1, 1)
        self.lbl_prob_user_1 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_1.setText("")
        self.lbl_prob_user_1.setObjectName("lbl_prob_user_1")
        self.gridLayout_6.addWidget(self.lbl_prob_user_1, 1, 2, 1, 1)
        self.lbl_user_22 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_22.setObjectName("lbl_user_22")
        self.gridLayout_6.addWidget(self.lbl_user_22, 8, 0, 1, 1)
        self.lbl_user_23 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_23.setObjectName("lbl_user_23")
        self.gridLayout_6.addWidget(self.lbl_user_23, 3, 0, 1, 1)
        self.lbl_user_17 = QtWidgets.QLabel(self.widget_3)
        self.lbl_user_17.setObjectName("lbl_user_17")
        self.gridLayout_6.addWidget(self.lbl_user_17, 5, 0, 1, 1)
        self.lbl_prob_user_6 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_6.setText("")
        self.lbl_prob_user_6.setObjectName("lbl_prob_user_6")
        self.gridLayout_6.addWidget(self.lbl_prob_user_6, 6, 2, 1, 1)
        self.lbl_prob_user_5 = QtWidgets.QLabel(self.widget_3)
        self.lbl_prob_user_5.setText("")
        self.lbl_prob_user_5.setObjectName("lbl_prob_user_5")
        self.gridLayout_6.addWidget(self.lbl_prob_user_5, 5, 2, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem12, 1, 1, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem13, 1, 3, 1, 1)
        self.horizontalLayout_4.addWidget(self.widget_3)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.gridLayout_9.addLayout(self.verticalLayout, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1291, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.chkBox_user_1, self.chkBox_user_2)
        MainWindow.setTabOrder(self.chkBox_user_2, self.chkBox_user_3)
        MainWindow.setTabOrder(self.chkBox_user_3, self.chkBox_user_4)
        MainWindow.setTabOrder(self.chkBox_user_4, self.chkBox_user_5)
        MainWindow.setTabOrder(self.chkBox_user_5, self.chkBox_user_6)
        MainWindow.setTabOrder(self.chkBox_user_6, self.chkBox_user_7)
        MainWindow.setTabOrder(self.chkBox_user_7, self.chkBox_user_8)
        MainWindow.setTabOrder(self.chkBox_user_8, self.chkBox_user_1)
        MainWindow.setTabOrder(self.chkBox_user_1, self.chkBox_user_2)
        MainWindow.setTabOrder(self.chkBox_user_2, self.chkBox_user_3)
        MainWindow.setTabOrder(self.chkBox_user_3, self.chkBox_user_4)
        MainWindow.setTabOrder(self.chkBox_user_4, self.chkBox_user_5)
        MainWindow.setTabOrder(self.chkBox_user_5, self.chkBox_user_6)
        MainWindow.setTabOrder(self.chkBox_user_6, self.chkBox_user_7)
        MainWindow.setTabOrder(self.chkBox_user_7, self.chkBox_user_8)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lbl_access.setText(_translate("MainWindow", "access"))
        self.lbl_acess_user_6.setText(_translate("MainWindow", "user 6"))
        self.lbl_acess_user_7.setText(_translate("MainWindow", "user 7"))
        self.lbl_acess_user_3.setText(_translate("MainWindow", "user 3"))
        self.lbl_acess_user_5.setText(_translate("MainWindow", "user 5"))
        self.lbl_acess_user_4.setText(_translate("MainWindow", "user 4"))
        self.lbl_acess_user_1.setText(_translate("MainWindow", "user 1"))
        self.lbl_acess_user_2.setText(_translate("MainWindow", "user 2"))
        self.lbl_acess_user_8.setText(_translate("MainWindow", "user 8"))
        self.label.setText(_translate("MainWindow", "choose how can acess:"))
        self.btn_record.setText(_translate("MainWindow", "start recording"))
        self.lbl_sentence_1.setText(_translate("MainWindow", "Open middle door"))
        self.label_4.setText(_translate("MainWindow", "sentence"))
        self.lbl_sentence_3.setText(_translate("MainWindow", "Grant me access"))
        self.label_3.setText(_translate("MainWindow", "Probability"))
        self.lbl_sentence_2.setText(_translate("MainWindow", "Unlock the gate"))
        self.lbl_user_24.setText(_translate("MainWindow", "user 1"))
        self.lbl_user_21.setText(_translate("MainWindow", "user 2"))
        self.lbl_user_18.setText(_translate("MainWindow", "user 7"))
        self.label_10.setText(_translate("MainWindow", "Probability"))
        self.lbl_user_19.setText(_translate("MainWindow", "user 4"))
        self.lbl_user_20.setText(_translate("MainWindow", "user 6"))
        self.label_9.setText(_translate("MainWindow", "user"))
        self.lbl_user_22.setText(_translate("MainWindow", "user 8"))
        self.lbl_user_23.setText(_translate("MainWindow", "user 3"))
        self.lbl_user_17.setText(_translate("MainWindow", "user 5"))
