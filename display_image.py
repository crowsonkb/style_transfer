import matplotlib.pyplot as plt


class ImageWindow:
    def __init__(self):
        self.imsh = None
        plt.ion()
        plt.show()

    def display(self, image):
        if self.imsh is None or not plt.fignum_exists(self.imsh.figure.number):
            self.imsh = plt.imshow(image, interpolation='nearest')
            self.imsh.axes.axis('off')
            self.imsh.figure.canvas.draw()
        else:
            self.imsh.set_data(image)
        plt.pause(1e-4)
