import whisper
import os
import time

from vllm.entrypoints.openai.api_server import transcription

# --- Configuration ---

# Choose the best model.
# "large-v3" is currently the largest and most accurate model.
# Be aware: It requires significant VRAM (around 10GB for GPU) or CPU power and RAM.
# If you encounter memory issues or it's too slow, try "large", then "medium".
model_name = "large-v3"

# Specify the language to improve accuracy, especially for non-English audio.
# "es" is the ISO 639-1 code for Spanish.
language = "es"

def get_transcription(audio_file_path, output_file):

    # --- Check if the audio file exists ---
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at '{audio_file_path}'")
        print("Please update the 'audio_file_path' variable to your actual MP3 file.")
        exit()

    # --- Load the Whisper model ---
    print(f"Loading Whisper model '{model_name}'. This may take some time for the first run...")
    start_load_time = time.time()
    try:
        model = whisper.load_model(model_name)
        print(f"Model loaded in {time.time() - start_load_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have enough RAM/VRAM and correct dependencies (like CUDA if using GPU).")
        exit()

    # --- Transcribe the audio ---
    print(f"\nStarting transcription of '{audio_file_path}' in Spanish...")
    start_transcribe_time = time.time()
    try:
        result = model.transcribe(audio_file_path, language=language, verbose=True)  # verbose=True shows progress
        transcription_text = result["text"]

        print(f"\nTranscription completed in {time.time() - start_transcribe_time:.2f} seconds.")
        print("\n--- Transcription ---")
        print("\n--------------------")
        with open(output_file, "w") as file:
            file.write(transcription_text)
        # Optionally, you can also access word-level segments and timestamps if needed:
        # print("\n--- Segments with Timestamps ---")
        # for segment in result["segments"]:
        #     print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Ensure your audio file is not corrupted and ffmpeg is correctly installed.")


def get_mp3_files_in_folder(folder_path):
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
            if os.path.isfile(full_path) and entry.lower().endswith(".mp3"):
                mp4_files.append(full_path)
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return mp4_files


if __name__ == "__main__":

    folder_path = './data/raw'
    mp3_files = get_mp3_files_in_folder(folder_path)

    for mp3_file in mp3_files:
        transcription_file =  mp3_file.replace('.mp3', '.txt')
        # Call the conversion function
        get_transcription(mp3_file, transcription_file)