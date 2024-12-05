import os
import cv2
import numpy as np


def convert_screenshots_to_videos(input_dir, output_dir, fps=30, clip_length=16):
    """
    Convert screenshot sequences to video clips

    Args:
    - input_dir: Directory with screenshot files
    - output_dir: Directory to save generated video clips
    - fps: Frames per second for output video
    - clip_length: Number of frames per video clip
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all screenshot files, sorted
    screenshots = sorted(
        [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg"))]
    )

    # Generate video clips
    for i in range(0, len(screenshots), clip_length):
        clip_screenshots = screenshots[i : i + clip_length]

        if len(clip_screenshots) == clip_length:
            # Create video writer
            output_path = os.path.join(
                output_dir, f"video_clip_{i//clip_length:04d}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (
                    cv2.imread(os.path.join(input_dir, clip_screenshots[0])).shape[1],
                    cv2.imread(os.path.join(input_dir, clip_screenshots[0])).shape[0],
                ),
            )

            # Write frames to video
            for screenshot in clip_screenshots:
                frame = cv2.imread(os.path.join(input_dir, screenshot))
                video_writer.write(frame)

            video_writer.release()


# Example usage
input_screenshots_dir = "../mcvd-pytorch/datasets/snake/imgs2"
output_videos_dir = "../mcvd-pytorch/datasets/snake/videos"
convert_screenshots_to_videos(input_screenshots_dir, output_videos_dir)
