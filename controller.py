import os


class Controller:

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.images = []
        self.current_image_index = 0

    def run(self, image_folder=None):
        print("[.] running application in ", image_folder)

        if image_folder is None or not os.path.isdir(image_folder):
            self.process_incorrect_directory()
            return

        self.images.clear()
        print("[-] looking for images")
        for item in os.listdir(image_folder):
            item_path = os.path.join(image_folder, item)
            if os.path.isfile(item_path) and self.is_image(item):
                print("[.] found image", item)

                self.images.append(Image.open(item_path))

        self.current_image_index = -1
        self.process_correct_directory(image_folder)
        self.process_next()