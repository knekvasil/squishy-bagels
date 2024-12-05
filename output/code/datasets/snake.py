import os
import torch
import torchvision.io as io
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SnakeDataset(Dataset):
    def __init__(
        self,
        data_path,
        train=True,
        transform=None,
        frames_per_sample=16,
        random_time=True,
        random_horizontal_flip=False,
    ):
        self.data_path = data_path
        self.transform = transform or transforms.ToTensor()
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time

        # Collect video file paths
        self.video_paths = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov")):
                    self.video_paths.append(os.path.join(root, file))

        # Split train/test
        split_idx = int(len(self.video_paths) * 0.8)
        self.video_paths = (
            self.video_paths[:split_idx] if train else self.video_paths[split_idx:]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Load video using torchvision.io
        video, audio, info = io.read_video(video_path)

        # Select frames
        if self.random_time:
            start_idx = torch.randint(
                0, max(1, video.shape[0] - self.frames_per_sample), (1,)
            ).item()
            frames = video[start_idx : start_idx + self.frames_per_sample]
        else:
            total_frames = video.shape[0]
            frame_indices = torch.linspace(
                0, total_frames - 1, self.frames_per_sample
            ).long()
            frames = video[frame_indices]

        frames = frames.float() / 255.0  # Normalize frames

        to_pil = transforms.ToPILImage()
        resize_transform = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()  # Convert PIL image to tensor after resizing

        # Apply resize and convert to tensor for each frame
        frames_transformed = torch.stack(
            [
                to_tensor(resize_transform(to_pil(frame.permute(2, 0, 1))))
                for frame in frames
            ],
            dim=0,
        )  # Stack frames along the time dimension

        # Generate labels for each frame based on the snake's directional movement
        labels = []
        for frame in frames:
            movement = self.get_snake_direction()
            labels.append(movement)

        # Pad the labels list if needed
        labels += [labels[-1]] * (self.frames_per_sample - len(labels))

        labels_tensor = torch.tensor(labels)  # Convert list to tensor

        return frames_transformed, labels_tensor

    # TODO: Placeholder for actual data (0, 1, 2) == (S, R, L)
    def get_snake_direction(self):
        return torch.randint(0, 3, (1,))
