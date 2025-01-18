import threading
import torch
from models.gen.edm import EDM
from models.gen.blocks import UNet
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from data.data import SequencesDataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import time
import random
import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, QSize

# Configuration
input_channels = 3
context_length = 4
actions_count = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 1

if torch.backends.mps.is_available():
    device = "mps"
ROOT_PATH = "../"


def local_path(path):
    return os.path.join(ROOT_PATH, path)


MODEL_PATH = local_path("models/model.pth")

edm = EDM(
    p_mean=-1.2,
    p_std=1.2,
    sigma_data=0.5,
    model=UNet(
        (input_channels) * (context_length + 1), 3, None, actions_count, context_length
    ),
    context_length=context_length,
    device=device,
)
edm.load_state_dict(torch.load(MODEL_PATH, map_location=device)["model"])

transform_to_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset = SequencesDataset(
    images_dir=local_path("training_data/snapshots"),
    actions_path=local_path("training_data/actions"),
    seq_length=context_length,
    transform=transform_to_tensor,
)


class State:
    def __init__(self):
        self.action = 0
        self.is_running = False
        self.frame_number = 0
        self.gen_imgs = None
        self.actions = None

    def reset(self):
        self.frame_number = 0
        self.is_running = False
        self.gen_imgs = None
        self.actions = None


state = State()

directions = {0: "Right", 1: "Left", 2: "Up", 3: "Down"}


# GUI Application
class SnakeGameApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.render_thread = None
        self.stop_event = threading.Event()

    def initUI(self):
        self.setWindowTitle("Snake Game Engine Diffusion Model")
        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)
        self.layout.addWidget(self.image_label, stretch=4)  # Give the image more space

        # Current direction label
        self.direction_label = QLabel("Current Direction: None", self)
        self.direction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.direction_label, stretch=1)

        self.button_layout = QHBoxLayout()
        self.left_button = QPushButton("Left", self)
        self.right_button = QPushButton("Right", self)
        self.up_button = QPushButton("Up", self)
        self.down_button = QPushButton("Down", self)
        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)

        self.button_layout.addWidget(self.left_button)
        self.button_layout.addWidget(self.right_button)
        self.button_layout.addWidget(self.up_button)
        self.button_layout.addWidget(self.down_button)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)

        self.layout.addLayout(self.button_layout, stretch=1)
        self.setLayout(self.layout)

        self.left_button.clicked.connect(lambda: self.on_button_click(1))
        self.right_button.clicked.connect(lambda: self.on_button_click(0))
        self.up_button.clicked.connect(lambda: self.on_button_click(2))
        self.down_button.clicked.connect(lambda: self.on_button_click(3))
        self.start_button.clicked.connect(self.start_rendering)
        self.stop_button.clicked.connect(self.stop_rendering)

    def on_button_click(self, input_action):
        state.action = input_action
        self.direction_label.setText(f"Current Direction: {directions[state.action]}")

    def start_rendering(self):
        if state.is_running:
            return
        state.reset()
        state.is_running = True
        self.stop_event.clear()

        self.render_thread = threading.Thread(target=self.render_loop)
        self.render_thread.start()

    def stop_rendering(self):
        state.reset()
        self.stop_event.set()
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join()
        self.image_label.clear()
        self.image_label.setText("Stopped rendering")
        self.direction_label.setText("Current Direction: None")

    def render_loop(self):
        while state.is_running and not self.stop_event.is_set():
            if state.frame_number >= 80:
                self.stop_rendering()
                return

            start_time = time.time()

            if state.frame_number == 0:
                index = random.randint(0, len(dataset) - 1)
                img, last_imgs, actions = dataset[index]
                state.gen_imgs = last_imgs.clone().to(device)
                state.actions = actions.to(device)

            state.actions = torch.concat(
                (state.actions, torch.tensor([state.action], device=device))
            )

            with torch.no_grad():
                gen_img = edm.sample(
                    10,
                    state.gen_imgs[0].shape,
                    state.gen_imgs[-context_length:].unsqueeze(0),
                    state.actions[-context_length:].unsqueeze(0),
                )[0]

            state.gen_imgs = torch.concat(
                [state.gen_imgs, gen_img[None, :, :, :]], dim=0
            )

            display_img = self.get_np_img(gen_img)
            display_img = np.ascontiguousarray(display_img)

            height, width, channel = display_img.shape
            bytes_per_line = channel * width
            qimage = QImage(
                display_img.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

            label_size = min(self.image_label.width(), self.image_label.height())
            scaled_pixmap = QPixmap.fromImage(qimage).scaled(
                label_size, label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)
            self.direction_label.setText(
                f"Current Direction: {directions[state.action]}"
            )

            state.frame_number += 1

            elapsed_time = time.time() - start_time
            delay = max(0, frame_time - elapsed_time)

            time.sleep(delay)

    def resizeEvent(self, event):
        """Override resizeEvent to scale the image when the window is resized."""
        super().resizeEvent(event)
        if state.is_running and state.gen_imgs is not None:
            gen_img = state.gen_imgs[-1]
            display_img = self.get_np_img(gen_img)
            display_img = np.ascontiguousarray(display_img)

            height, width, channel = display_img.shape
            bytes_per_line = channel * width
            qimage = QImage(
                display_img.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

            label_size = min(self.image_label.width(), self.image_label.height())
            scaled_pixmap = QPixmap.fromImage(qimage).scaled(
                label_size, label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)

    def get_np_img(self, tensor: torch.Tensor) -> np.ndarray:
        return (
            (tensor * 127.5 + 127.5)
            .long()
            .clip(0, 255)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )


FPS = 0.2
frame_time = 1 / FPS

# Run application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SnakeGameApp()
    ex.show()
    sys.exit(app.exec())
