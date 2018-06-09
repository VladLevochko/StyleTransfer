import sys
from PyQt5.QtWidgets import QApplication
from model import Classifier
from base_style_transfer import StyleTransferBase
from view import QtView
from controller import Controller


if __name__ == "__main__":

    app = QApplication(sys.argv)

    model = StyleTransferBase()
    view = QtView()
    controller = Controller(model, view)

    controller.run()

    sys.exit(app.exec_())
