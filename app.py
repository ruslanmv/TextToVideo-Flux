import os
import multiprocessing
import subprocess
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from min_dalle import MinDalle
from moviepy.editor import VideoFileClip
import moviepy.editor as mpy
from PIL import Image, ImageDraw, ImageFont
from mutagen.mp3 import MP3
from gtts import gTTS
from pydub import AudioSegment
import textwrap
import gradio as gr
import matplotlib.pyplot as plt
import gc  # Garbage collector
from huggingface_hub import snapshot_download
from audio import *

# Ensure proper multiprocessing start method
multiprocessing.set_start_method("spawn", force=True)

# GPU Fallback Setup
if os.environ.get("SPACES_ZERO_GPU") is not None:
    import spaces
else:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=None):
            def wrapper(fn):
                return fn
            return wrapper if func is None else wrapper(func)

# Download necessary NLTK data
def setup_nltk():
    """Ensure required NLTK data is available."""
    try:
        nltk.download('punkt_tab')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

setup_nltk()

# Constants
DESCRIPTION = (
    "Video Story Generator with Audio\n"
    "PS: Generation of video by using Artificial Intelligence via dalle-mini, distilbart, and GTTS."
)
TITLE = "Video Story Generator with Audio by using dalle-mini, distilbart, and GTTS."

# Load Tokenizer and Model for Text Summarization
def load_text_summarization_model():
    """Load the tokenizer and model for text summarization."""
    print("Loading text summarization model...")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_text_summarization_model()

# Log GPU Memory (optional, for debugging)
def log_gpu_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        print(subprocess.check_output('nvidia-smi').decode('utf-8'))
    else:
        print("CUDA is not available. Cannot log GPU memory.")

# Check GPU Availability
def check_gpu_availability():
    """Print GPU availability and device details."""
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(torch.cuda.get_device_properties(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Running on CPU.")

check_gpu_availability()

# GPU-Safe MinDalle Model Loading
def initialize_min_dalle():
    """Load the MinDalle model with GPU support."""
    if torch.cuda.is_available():
        @spaces.GPU(duration=60 * 3)
        def load_model():
            print("Loading MinDalle model on GPU...")
            return MinDalle(
                is_mega=True,
                models_root='pretrained',
                is_reusable=False,
                is_verbose=True,
                dtype=torch.float16,
                device='cuda'
            )
        return load_model()
    else:
        print("Loading MinDalle model on CPU...")
        return MinDalle(
            is_mega=True,
            models_root='pretrained',
            is_reusable=False,
            is_verbose=True,
            dtype=torch.float32,
            device='cpu'
        )





def generate_image_with_min_dalle(
    model: MinDalle,
    text: str,
    seed: int = -1,
    grid_size: int = 1
):
    """
    Generates an image from text using MinDalle.

    Args:
        model: The preloaded MinDalle model.
        text: The text prompt to generate the image from.
        seed: The random seed for image generation. -1 for random.
        grid_size: The grid size for multiple image generation.

    Returns:
        A PIL Image object.
    """
    print(f"DEBUG: Generating image with MinDalle for text: '{text}'")
    model.is_reusable = False
    with torch.no_grad():
        image = model.generate_image(
            text,
            seed,
            grid_size,
            is_verbose=False
        )

    # Clear GPU memory after generation
    torch.cuda.empty_cache()
    gc.collect()

    print("DEBUG: Image generated successfully.")
    return image


# --------- End of MinDalle Functions ---------
# Merge audio files

from pydub import AudioSegment
import os


# Initialize MinDalle Model
min_dalle_model = initialize_min_dalle()


# Function to generate video from text
def get_output_video(text):
    print("DEBUG: Starting get_output_video function...")
  
   # Summarize the input text
    print("DEBUG: Summarizing text...")
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    summary_ids = model.generate(inputs["input_ids"])
    summary = tokenizer.batch_decode(
        summary_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    plot = list(summary[0].split('.'))
    print(f"DEBUG: Summary generated: {plot}")

    # Generate images for each sentence in the plot
    generated_images = []
    for i, senten in enumerate(plot[:-1]):
        print(f"DEBUG: Generating image {i+1} of {len(plot)-1}...")
        image_dir = f"image_{i}"
        os.makedirs(image_dir, exist_ok=True)
        image = generate_image_with_min_dalle(
            min_dalle_model,
            text=senten,
            seed=1,
            grid_size=1
        )
        generated_images.append(image)
        image_path = os.path.join(image_dir, "generated_image.png")
        image.save(image_path)
        print(f"DEBUG: Image generated and saved to {image_path}")

        #del min_dalle_model
        torch.cuda.empty_cache()
        gc.collect()

    # Create subtitles from the plot
    sentences = plot[:-1]
    print("DEBUG: Creating subtitles...")
    assert len(generated_images) == len(sentences), "Mismatch in number of images and sentences."
    sub_names = [nltk.tokenize.sent_tokenize(sentence) for sentence in sentences]

    # Add subtitles to images with dynamic adjustments
    def get_dynamic_wrap_width(font, text, image_width, padding):
        # Estimate the number of characters per line dynamically
        avg_char_width = sum(font.getbbox(c)[2] for c in text) / len(text)
        return max(1, (image_width - padding * 2) // avg_char_width)

    def draw_multiple_line_text(image, text, font, text_color, text_start_height, padding=10):
        draw = ImageDraw.Draw(image)
        image_width, _ = image.size
        y_text = text_start_height
        lines = textwrap.wrap(text, width=get_dynamic_wrap_width(font, text, image_width, padding))
        for line in lines:
            line_width, line_height = font.getbbox(line)[2:]
            draw.text(((image_width - line_width) / 2, y_text), line, font=font, fill=text_color)
            y_text += line_height + padding

    def add_text_to_img(text1, image_input):
        print(f"DEBUG: Adding text to image: '{text1}'")
        # Scale font size dynamically
        base_font_size = 30
        image_width, image_height = image_input.size
        scaled_font_size = max(10, int(base_font_size * (image_width / 800)))  # Adjust 800 based on typical image width
        path_font = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if not os.path.exists(path_font):
            path_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(path_font, scaled_font_size)

        text_color = (255, 255, 0)
        padding = 10

        # Estimate starting height dynamically
        line_height = font.getbbox("A")[3] + padding
        total_text_height = len(textwrap.wrap(text1, get_dynamic_wrap_width(font, text1, image_width, padding))) * line_height
        text_start_height = image_height - total_text_height - 20

        draw_multiple_line_text(image_input, text1, font, text_color, text_start_height, padding)
        return image_input


    # Process images with subtitles
    generated_images_sub = []
    for k, image in enumerate(generated_images):
        text_to_add = sub_names[k][0]
        result = add_text_to_img(text_to_add, image.copy())
        generated_images_sub.append(result)
        result.save(f"image_{k}/generated_image_with_subtitles.png")



    # Generate audio for each subtitle
    mp3_names = []
    mp3_lengths = []
    for k, text_to_add in enumerate(sub_names):
        print(f"DEBUG: Generating audio for: '{text_to_add[0]}'")
        f_name = f'audio_{k}.mp3'
        mp3_names.append(f_name)
        myobj = gTTS(text=text_to_add[0], lang='en', slow=False)
        myobj.save(f_name)
        audio = MP3(f_name)
        mp3_lengths.append(audio.info.length)
        print(f"DEBUG: Audio duration: {audio.info.length} seconds")

    # Merge audio files
    export_path = merge_audio_files(mp3_names)

    # Create video clips from images
    clips = []
    for k, img in enumerate(generated_images_sub):
        duration = mp3_lengths[k]
        print(f"DEBUG: Creating video clip {k+1} with duration: {duration} seconds")
        clip = mpy.ImageClip(f"image_{k}/generated_image_with_subtitles.png").set_duration(duration + 0.5)
        clips.append(clip)

    # Concatenate video clips
    print("DEBUG: Concatenating video clips...")
    concat_clip = mpy.concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile("result_no_audio.mp4", fps=24)

    # Combine video and audio
    movie_name = 'result_no_audio.mp4'
    movie_final = 'result_final.mp4'

    def combine_audio(vidname, audname, outname, fps=24):
        print(f"DEBUG: Combining audio for video: '{vidname}'")
        my_clip = mpy.VideoFileClip(vidname)
        audio_background = mpy.AudioFileClip(audname)
        final_clip = my_clip.set_audio(audio_background)
        final_clip.write_videofile(outname, fps=fps)

    combine_audio(movie_name, export_path, movie_final)

    # Clean up
    print("DEBUG: Cleaning up files...")
    for i in range(len(generated_images_sub)):
        shutil.rmtree(f"image_{i}")
        os.remove(f"audio_{i}.mp3")
    os.remove("result.mp3")
    os.remove("result_no_audio.mp4")

    print("DEBUG: Cleanup complete.")
    print("DEBUG: get_output_video function completed successfully.")
    return 'result_final.mp4'

# Example text (can be changed by user in Gradio interface)
text = 'Once, there was a girl called Laura who went to the supermarket to buy the ingredients to make a cake. Because today is her birthday and her friends come to her house and help her to prepare the cake.'

# Create Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# Video Generator from stories with Artificial Intelligence")
    gr.Markdown("A story can be input by user. The story is summarized using DistilBART model. Then, the images are generated by using Dalle-mini, and the subtitles and audio are created using gTTS. These are combined to generate a video.")
    with gr.Row():
        with gr.Column():
            input_start_text = gr.Textbox(value=text, label="Type your story here, for now a sample story is added already!")
            with gr.Row():
                button_gen_video = gr.Button("Generate Video")
        with gr.Column():
            output_interpolation = gr.Video(value="test.mp4", label="Generated Video")  # Set default video
    gr.Markdown("<h3>Future Works </h3>")
    gr.Markdown("This program is a text-to-video AI software generating videos from any prompt! AI software to build an art gallery. The future version will use Dalle-2. For more info visit [ruslanmv.com](https://ruslanmv.com/) ")
    button_gen_video.click(fn=get_output_video, inputs=input_start_text, outputs=output_interpolation)

# Launch the Gradio app
demo.launch(debug=True, share=True)
