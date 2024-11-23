import os
import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image, ImageChops
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
from tqdm import tqdm
from uuid import uuid1

# Check CUDA availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.set_float32_matmul_precision(["high", "highest"][0])

# Load the model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0", trust_remote_code=True
)
birefnet.to(device)
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    image = process(im)
    image_path = os.path.join(output_folder, "no_bg_image.png")
    image.save(image_path)
    return (image, origin), image_path

@spaces.GPU
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def process_file(f):
    name_path = f.rsplit(".",1)[0]+".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

def remove_background(image):
    """Remove background from a single image."""
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Convert the prediction to a mask
    mask = (pred * 255).byte()  # Convert to 0-255 range
    mask_pil = transforms.ToPILImage()(mask).convert("L")
    mask_resized = mask_pil.resize(image.size, Image.LANCZOS)

    # Apply the mask to the image
    image.putalpha(mask_resized)

    return image, mask_resized

def process_video(input_video_path):
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
    output_video_path = os.path.join(output_folder, "no_bg_video.mp4")
    processed_clip = ImageSequenceClip(frames, fps=video_clip.fps)
    processed_clip.write_videofile(output_video_path, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuva420p'])

    return output_video_path

# Gradio components
slider1 = ImageSlider(label="RMBG-2.0", type="pil")
slider2 = ImageSlider(label="RMBG-2.0", type="pil")
image = gr.Image(label="Upload an image")
image2 = gr.Image(label="Upload an image", type="filepath")
text = gr.Textbox(label="Paste an image URL")
png_file = gr.File(label="output png file")
video_input = gr.Video(label="Upload a video")
video_output = gr.Video(label="Processed video")

# Example videos
example_videos = [
    "pexels-cottonbro-5319934.mp4",
    "300_A_car_is_running_on_the_road.mp4",
    "A_Terracotta_Warrior_is_skateboarding_9033688.mp4"
]

# Gradio interfaces
tab1 = gr.Interface(
    fn, inputs=image, outputs=[slider1, gr.File(label="output png file")], examples=[load_img("giraffe.jpg", output_type="pil")], api_name="image"
)

tab2 = gr.Interface(fn, inputs=text, outputs=[slider2, gr.File(label="output png file")], examples=["http://farm9.staticflickr.com/8488/8228323072_76eeddfea3_z.jpg"], api_name="text")
#tab3 = gr.Interface(process_file, inputs=image2, outputs=png_file, examples=["giraffe.jpg"], api_name="png")
tab4 = gr.Interface(process_video, inputs=video_input, outputs=video_output, examples=example_videos, api_name="video", cache_examples = False)

# Gradio tabbed interface
demo = gr.TabbedInterface(
    [tab4, tab1, tab2], ["input video", "input image", "input url"], title="RMBG-2.0 for background removal"
)

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
