'''
BESKRIVELSE
Dette programmet tar inn et datasett fra en mappe, augmenterer dataen, og legger augmentert data inn i ny mappe.
Github: https://github.com/musikalkemist/audioDataAugmentationTutorial/tree/main/5
Youtube: https://www.youtube.com/watch?v=rPj9_rIBqNA&t=508s


Sist oppdatert: 13.12.2024
'''

import librosa
import soundfile as sf
import os
import random  # Importer random-modulen for seed-setting
from audiomentations import Compose, PitchShift, LowPassFilter

_INPUT_FOLDER = ''
SEED = 3 # Velg en seed-verdi for å gjøre resultatet reproduserbart
_OUTPUT_FOLDER = f'_{SEED}'

# Raw audio augmentation uten seed i konstruktøren
augment_raw_audio = Compose(
    [
        #LowPassFilter(min_cutoff_freq=, max_cut_off_freq=, p=1),
        PitchShift(min_semitones=-3.0, max_semitones=3.0, p=1),
    ]
)

# Funksjon for å samle alle .wav-filer i datasettet og finne kategoriene automatisk
def get_wav_files_and_labels(dataset_path):
    files = []  # Filbanen til alle filene
    labels = []  # Tilhørende klasse
    categories = next(os.walk(dataset_path))[1]  # Henter mappenavnene som representerer kategorier

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                files.append(os.path.join(category_path, file))
                labels.append(category)  # Label er kategorien
    return files, labels, categories

# Hovedprogram for å augmentere filer
if __name__ == '__main__':
    # Hent filer, labels, og kategorier fra treningsmappen
    wav_files, labels, categories = get_wav_files_and_labels(_INPUT_FOLDER)

    # Opprett hovedutgangsmappen med seed i navnet, hvis den ikke finnes
    os.makedirs(_OUTPUT_FOLDER, exist_ok=True)

    # Opprett kategorimapper i utgangsmappen
    for category in categories:
        category_output_path = os.path.join(_OUTPUT_FOLDER, category)
        os.makedirs(category_output_path, exist_ok=True)

    # Sett seed for å sikre reproduserbare resultater
    random.seed(SEED)

    # Gå gjennom lydfilene og augmenter dem
    for wav_file, label in zip(wav_files, labels):
        # Last inn original lydfil
        signal, sr = librosa.load(wav_file, sr=None)  # Bruker original samplerate

        # Augmenter lydfilen
        augmented_signal = augment_raw_audio(signal, sr)

        # Sett opp sti for å lagre augmentert fil
        filename = os.path.basename(wav_file)
        augmented_wav_file = os.path.join(_OUTPUT_FOLDER, label, f"PS_{filename}")

        # Lagre augmentert lydfil
        sf.write(augmented_wav_file, augmented_signal, sr)
        print(f'Ferdig augmentering av fil: {augmented_wav_file}')


