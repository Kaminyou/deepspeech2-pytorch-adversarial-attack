import librosa    
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O parameters
    parser.add_argument('--input_folder', type=str, help='folder with original wav files')
    parser.add_argument('--output_folder', type=str, help='folder to save the resampled wav files')
    args = parser.parse_args()

    for wav_file in os.listdir(args.input_folder):
        if wav_file[-4:] == ".wav":
            print(f"Convert {wav_file}")
            y, s = librosa.load(os.path.join(args.input_folder, wav_file), sr=16000)
            librosa.output.write_wav(os.path.join(args.output_folder, wav_file), y, s)