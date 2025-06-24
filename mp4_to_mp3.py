from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import whisper
import os
import time


def mp4_to_mp3(input_mp4_path, output_mp3_path):
    """
    Converts an MP4 video file to an MP3 audio file.

    Args:
        input_mp4_path (str): The path to the input MP4 file.
        output_mp3_path (str): The path where the output MP3 file will be saved.
    """
    try:
        # Load the video file
        video_clip = VideoFileClip(input_mp4_path)

        # Extract the audio from the video clip
        audio_clip = video_clip.audio

        # Write the audio to an MP3 file
        audio_clip.write_audiofile(output_mp3_path, codec='mp3')

        # Close the clips to free resources
        audio_clip.close()
        video_clip.close()

        print(f"Successfully converted '{input_mp4_path}' to '{output_mp3_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_mp4_files_in_folder(folder_path):
    """
    Reads all MP4 files in a specified folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of full paths to the MP4 files found.
    """
    mp4_files = []
    try:
        # List all entries (files and subdirectories) in the given folder
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            # Check if it's a file and ends with .mp4 (case-insensitive)
            if os.path.isfile(full_path) and entry.lower().endswith(".mp4"):
                mp4_files.append(full_path)
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return mp4_files

if __name__ == "__main__":

    folder_path = './data/raw'
    mp4_files = get_mp4_files_in_folder(folder_path)

    for mp4_file in mp4_files:
        mp3_file = mp4_file.replace('.mp4', '.mp3')
        # Call the conversion function
        mp4_to_mp3(mp4_file, mp3_file)

