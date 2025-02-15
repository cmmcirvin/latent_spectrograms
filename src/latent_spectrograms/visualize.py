import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
from skimage.transform import resize


class Visualizer:
    """
    Generates visual representations of the 2D data with hovering spectrograms.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, iq_data: np.ndarray, text_labels: list
    ):
        """
        Initializes the Visualizer object.

        Parameters:
        x (np.ndarray): x-axis data
        y (np.ndarray): y-axis data
        iq_data (np.ndarray): IQ data for generating spectrograms
        text_labels (list): Labels for the data points
        """
        self.x = x
        self.y = y
        self.text_labels = text_labels
        self.iq_data = iq_data

    def visualize(self, img_shape=(100, 100), cmap="viridis", title="Latent space"):
        """
        Plots the 2D data with hoverable spectrograms.

        Parameters:
        img_shape (tuple): The shape of the spectrogram images.
        cmap (str): The colormap used for the scatter plot and spectrograms.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        arr = np.empty((len(self.iq_data), img_shape[0], img_shape[1]))
        for i in range(len(self.iq_data)):
            spectrum, _, _, _ = plt.specgram(
                self.iq_data[i], 256, 1.0, noverlap=254, visible=False
            )
            arr[i, :, :] = resize(spectrum, img_shape, anti_aliasing=True)

        unique_labels = list(set(self.text_labels))
        labels = [unique_labels.index(label) for label in self.text_labels]
        collection = ax.scatter(self.x, self.y, cmap=cmap, c=labels)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_ylim(self.y.min(), self.y.max())

        im = OffsetImage(arr[0, :, :], cmap=cmap)
        im_ab = AnnotationBbox(
            im,
            (0, 0),
            xycoords="data",
            boxcoords="offset points",
        )
        ax.add_artist(im_ab)
        im_ab.set_visible(False)

        txt = TextArea("")
        txt_ab = AnnotationBbox(
            txt,
            (0, 0),
            xycoords="data",
            boxcoords="offset points",
        )
        ax.add_artist(txt_ab)
        txt_ab.set_visible(False)

        def hover(event):
            if collection.contains(event)[0]:
                ind = collection.contains(event)[1]["ind"][0]

                im_ab.xybox = (0.0, 0.0)
                im_ab.xy = (self.x[ind], self.y[ind])
                im_ab.set_visible(True)

                txt_ab.xybox = (0.0, -img_shape[1] // 2)
                txt_ab.xy = (self.x[ind], self.y[ind])
                txt_ab.set_visible(True)

                im.set_data(arr[ind, :, :])
                txt.set(text=self.text_labels[ind])
            else:
                im_ab.set_visible(False)
                txt_ab.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.title(title)
        plt.show()
