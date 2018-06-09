import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QGridLayout, QLineEdit, QPushButton, \
    QFormLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PIL.ImageQt import ImageQt


class QtView(QWidget):
    def __init__(self, title="Image Classifier"):
        super().__init__()
        self.title = title
        self.top = 0
        self.left = 0
        # self.width = 1000
        # self.height = 500
        self.controller = None

        self.layout = None
        self.image_label = None
        self.grid = None
        self.grid_labels = []
        self.state_label = None
        self.path_input = None
        self.process_button = None
        self.next_button = None
        self.prev_button = None

        self.max_image_width = 500
        self.max_image_height = 500

        self.font = QFont("Arial", 20, QFont.Bold)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.get_image_widget())
        self.layout.addWidget(self.get_right_side_widget())
        self.setLayout(self.layout)

        self.show()

