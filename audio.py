import imageio_ffmpeg as ffmpeg
import subprocess
import os

def merge_audio_files(mp3_names, silence_duration=500, output_path='result.mp3'):
    """
    Merges a list of MP3 files into a single audio file with silence in between.

    Args:
        mp3_names (list): List of MP3 file paths to merge.
        silence_duration (int): Duration of silence (in milliseconds) between audio segments.
        output_path (str): Path to save the resulting merged audio.

    Returns:
        str: Path to the resulting merged audio file.
    """
    print(f"DEBUG: mp3_names: '{mp3_names}'")

    # Get the FFmpeg executable path from imageio_ffmpeg
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    print(f"DEBUG: Using FFmpeg executable at: {ffmpeg_path}")

    # Validate input files
    for mp3_file in mp3_names:
        if not os.path.exists(mp3_file):
            raise FileNotFoundError(f"Audio file '{mp3_file}' not found.")

    # Generate silence file using FFmpeg
    silence_file = "silence.mp3"
    silence_duration_seconds = silence_duration / 1000  # Convert to seconds
    silence_cmd = [
        ffmpeg_path,
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(silence_duration_seconds),
        "-q:a", "9",  # Set quality for silence (0 is best, 9 is worst)
        silence_file
    ]
    print("DEBUG: Generating silence file...")
    subprocess.run(silence_cmd, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Create a temporary file to define input list for FFmpeg
    concat_list_path = "concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for i, mp3_file in enumerate(mp3_names):
            # Use absolute paths for FFmpeg compatibility
            f.write(f"file '{os.path.abspath(mp3_file)}'\n")
            if i < len(mp3_names) - 1:  # Add silence between files, except after the last
                f.write(f"file '{os.path.abspath(silence_file)}'\n")

    # Merge audio files with silence and re-encode to ensure compatibility
    try:
        merge_cmd = [
            ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:a", "libmp3lame",  # Re-encode to MP3 using libmp3lame
            "-q:a", "2",  # Set quality level (0 is best, 9 is worst)
            output_path
        ]
        print("DEBUG: Merging audio files...")
        print(f"DEBUG: FFmpeg command: {' '.join(merge_cmd)}")
        result = subprocess.run(merge_cmd, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"DEBUG: FFmpeg stdout: {result.stdout}")
        print(f"DEBUG: FFmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"DEBUG: FFmpeg error output: {e.stderr}")
        print(f"DEBUG: FFmpeg standard output: {e.stdout}")
        raise RuntimeError(f"FFmpeg merging failed: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
        if os.path.exists(silence_file):
            os.remove(silence_file)

    print('DEBUG: Audio merging complete!')
    return output_path

# Test the function with audio_0.mp3 and audio_1.mp3
if __name__ == "__main__":
    mp3_files = ["audio_0.mp3", "audio_1.mp3"]  # Ensure these files exist in your directory
    try:
        output_file = merge_audio_files(mp3_files, silence_duration=500, output_path="merged_audio_result.mp3")
        print(f"Audio files merged successfully. Output saved at: {output_file}")
    except Exception as e:
        print(f"An error occurred during the merging process: {e}")
