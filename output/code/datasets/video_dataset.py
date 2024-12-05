import os
import torch
import torchvision.io as io
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoDataset(Dataset):
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
            # Randomly sample frames
            start_idx = torch.randint(
                0, max(1, video.shape[0] - self.frames_per_sample), (1,)
            ).item()
            frames = video[start_idx : start_idx + self.frames_per_sample]
        else:
            # Evenly sample frames
            total_frames = video.shape[0]
            frame_indices = torch.linspace(
                0, total_frames - 1, self.frames_per_sample
            ).long()
            frames = video[frame_indices]

        # Normalize frames (convert to float and scale)
        frames = frames.float() / 255.0

        # Apply transforms
        frames = torch.stack(
            [self.transform(frame.permute(2, 0, 1)) for frame in frames]
        )

        return frames
