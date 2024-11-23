from PIL import Image, ImageChops
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
from tqdm import tqdm
from uuid import uuid1
import os

# Load the model
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision('high')  # Set precision
model.to('cuda')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image):
    """Remove background from a single image."""
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Convert the prediction to a mask
    mask = (pred * 255).byte()  # Convert to 0-255 range
    mask_pil = transforms.ToPILImage()(mask).convert("L")
    mask_resized = mask_pil.resize(image.size, Image.LANCZOS)

    # Apply the mask to the image
    image.putalpha(mask_resized)

    return image, mask_resized

def process_video(input_video_path, output_video_path):
    """Process a video to remove the background from each frame."""
    # Load the video
    video_clip = VideoFileClip(input_video_path)

    # Process each frame
    frames = []
    for frame in tqdm(video_clip.iter_frames()):
        frame_pil = Image.fromarray(frame)
        frame_no_bg, mask_resized = remove_background(frame_pil)
        path = "{}.png".format(uuid1())
        frame_no_bg.save(path)
        frame_no_bg = Image.open(path).convert("RGBA")
        os.remove(path)

        # Convert mask_resized to RGBA mode
        mask_resized_rgba = mask_resized.convert("RGBA")

        # Apply the mask using ImageChops.multiply
        output = ImageChops.multiply(frame_no_bg, mask_resized_rgba)
        output_np = np.array(output)
        frames.append(output_np)

    # Save the processed frames as a new video
    processed_clip = ImageSequenceClip(frames, fps=video_clip.fps)
    processed_clip.write_videofile(output_video_path, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuva420p'])

if __name__ == "__main__":
    from IPython import display
    # Example usage
    input_video_path = "300_A_car_is_running_on_the_road.mp4"  # Replace with your video path
    output_video_path = "300_A_car_is_running_on_the_road_no_bg.mp4"
    process_video(input_video_path, output_video_path)
    display.Video("300_A_car_is_running_on_the_road_no_bg.mp4")
    pass
