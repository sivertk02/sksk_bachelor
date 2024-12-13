'''BESKRIVELSE
Dette programmet tar inn et datasett med lyfiler og gjør de om til 3 sekunders segmenter. Restene blir klippet bort
Kode er generert av Chat GPT

Valg av samperate og segmentlenkde er inspirert av:
samme som Irfran i DeepShip: an Underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification
Sist oppdatert: 13.12.2024
'''


import os
import librosa
import soundfile as sf

DATASET_PATH = ''
OUTPUT_PATH = ''

SAMPLE_RATE = 16000 # Bruker 16khz
SEGMENT_DURATION = 3  # 3 sekunder
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

def process_audio_files(dataset_path, output_path, sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            # Ekstrakt kategorien fra mappe
            dirpath_components = dirpath.split("/")
            label = dirpath_components[-1]

            # Opprett mappe for å lagre segmentene
            label_output_path = os.path.join(output_path, label)
            if not os.path.exists(label_output_path):
                os.makedirs(label_output_path)

            print(f"Processing category: {label}")
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                try:
                    signal, sr = librosa.load(file_path, sr=sample_rate)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                # Finn antall hele segmenter (3 sekunder)
                num_segments = len(signal) // SAMPLES_PER_SEGMENT

                for s in range(num_segments):
                    start_sample = s * SAMPLES_PER_SEGMENT
                    end_sample = start_sample + SAMPLES_PER_SEGMENT
                    
                    # Hent ut segmentet
                    segment = signal[start_sample:end_sample]

                    # Lagre segmentet som en ny lydfil
                    segment_filename = f"{f[:-4]}_segment{s}.wav"  # Gir hver segmentfil et unikt navn
                    segment_path = os.path.join(label_output_path, segment_filename)
                    
                    # Lagre segmentet til fil
                    sf.write(segment_path, segment, sample_rate)

                print(f"Processed {f} into {num_segments} segments.")

if __name__ == "__main__":
    process_audio_files(DATASET_PATH, OUTPUT_PATH)








