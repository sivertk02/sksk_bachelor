'''
Dette programmet tar inn en lydfil og augmenterer den ved bruk av tidsavhengig 
frekvensforskyvning
ChatGPT er blitt brukt til å generere deler av programmet

Fremgangsmåte er inspiert av artikkel:
https://download.ssrn.com/apac/258ce311-10ac-436e-8b5c-b1793c03984a-meca.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCrvH4YVGE5%2FON7iJBufbLv3yCfE8TMnWJ4wjxEdmXOUAIhAKlBuSOcM7wT3jXpoEMruyjWV8mP1vx5NRmPfycaS%2B2EKr0FCHQQBBoMMzA4NDc1MzAxMjU3IgxqDaty2yp4uCSl%2FxYqmgUHE85k16eJXfcQqRc%2F5luhKU74%2Bi0Wi%2BsLPOZRO9lTtvVLxp0ZgbTXSiGUlxQgY3Gw5UjZuwwyUuNoi8Cx2HP9UpeYFLslI4JAOfqdokQUbyEzsic66Fu05vBf%2FhFH7WQiVIoiHVpj8lUPZI%2Fl2oP%2FIct7dPASgI6k%2FYtL8TbtS7WGQRXMzitTfEP0EKuBeq2sTCtdOnFomKVbDHMBSgiIbSWF8n8RRhIH5bfZO0Ne%2BdSwPDKysu5cvBu6fY6aMVvz%2FsRmVUPwsgOblpmCVo42gKXcHq74%2Fo3YKc2FkgDEtIeAcBb%2ByAm4EjVac3u1DhA4ywn2IAP7yRfGFL6HtG9dm0Z9oKd2jPskPZpHM6lV%2FuB2N0Z0cLbK7TNkro31Z4OtjNPsj%2F6rrwT5Z9Oo0EpJpp2ZXsY9VJgv6z5nSeLUWVh5FfVK38L%2BMeTWIz7vUNxQrTImKe4HgGm8e5RPk88nQzWLXgrjeaAUtUFwrFpFzaXb4ByWfdGjEHUu2Zvntw4%2BoSnU%2BnYBJiWsqZZ4opl7%2BhCQ78d78w0pkO9R%2FPGafLuWH68oBBFsmlGFzm8jelcWpHxiw9%2FEq8asJz0Evo9kQg8%2FSV%2F6nQsL7ND6i5mfZ%2BCEqob76cCw5qXfBSAQQAHqcPGdDdmbboqGZlU30OZsOhuH0xlFGskpz7%2FzrIXy1D0paSQeyvDJHcgsh41sTxbl69U1AHBoqjVgY1bY6zmNXo9UBkntbHQsUb%2BhLS1dcXmtAVJltuKniwwwO2hrlUrg1DtEUhwZWnAahSBVUACaycfNtu9TkgURTkMDDnFlo01X0eG88w6d3IgsD4%2FT3RbRkwUudj8lIoi2WJi2uS%2F7haRKClVwR%2FRLHDtTOJzVvmrCg9I9YytpDbEw9bamugY6sAGsgdlpBqkmMXxRm2QcxCjzCLeQj6nMEGWvFxJpvP1Kka4AyOJCX5ysx7UDjH9GMuUhAa26DGm3atLJZ0eEa%2BJew0aU7fgnJddW5dL1RbB7h3GaHH0TmTAhErwbOoO61wDYlDLjL3e8oeBKxdnycCV%2F3d9yeSTf%2FsUVSauEHVF1%2BfKyw9uyVZz4OZ2ApWQVKBq0ZQQsL34nugPVy1MZWhcoV%2F5Ii8m2BWMieOf9MYczYg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241129T115132Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWEWYH5DSBZ%2F20241129%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=9782467a5c9daa7fc567a54ee6821caabad93b45af629d628d57aa405c81203d&abstractId=4347042

Sist oppdatert: 13.12.2024
'''

from librosa import get_duration, load
from numpy import arange, where, linspace, clip, cos ,pi
from random import choice, seed
from itertools import accumulate
from scipy.interpolate import interp1d
from soundfile import write
from typing import Tuple
import os

_SEED: int = 1
_INPUT_FOLDER: str = ''
_OUTPUT_FOLDER: str = f'_{_SEED}'
_SAMPLE_RATE: int = 16000


def ips(input_path:str,
        sample_rate:int = _SAMPLE_RATE | 16000,
        A: float = 1.02) -> Tuple[float, float]:
    '''Augments audiofile with Improved Pitch Shifting'''

    # Hent signaldata
    try:
        
        duration = get_duration(path=input_path, sr=sample_rate)
        amplitude_signal, _ = load(path=input_path, sr=sample_rate)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return
    
    num_samples = len(amplitude_signal) 
    
    # IPS
    period: float = duration # Perioden til cosinus
    PHASE_CHOICES = [0, pi / 2, pi, 3 * pi / 2]
    start_phase = choice(PHASE_CHOICES)

    # Shifting samplerate for each sample: fs(n)
    n_values = arange(num_samples)
    fs = sample_rate * A ** cos((2 * pi * n_values * duration) / (period * num_samples) + start_phase)

    # Time intervall for each sample: Ts(n)
    Ts = where(fs != 0, 1 / fs, 0)

    # Calculation interpolar moment for each sample: sum(dt)
    ts = list(accumulate(Ts))

    # Interpolates new amplitude values for constant
    ts = clip(ts, 0, duration)
    new_time = linspace(0, duration, num_samples)
    interpolator = interp1d(ts, amplitude_signal, kind='linear', fill_value="extrapolate")
    new_amplitude = interpolator(new_time)

    return new_amplitude, sample_rate

def get_wav_files_and_labels(dataset_path):
    '''Funksjon for å samle alle .wav-filer i datasettet og finne kategoriene automatisk'''
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
    seed(_SEED)

    # Gå gjennom lydfilene og augmenter dem
    for wav_file, label in zip(wav_files, labels):
        # Augmenter lydfilen
        augmented_signal, sample_rate = ips(input_path=wav_file, sample_rate=_SAMPLE_RATE)

        # Sett opp sti for å lagre augmentert fil
        filename = os.path.basename(wav_file)
        augmented_wav_file = os.path.join(_OUTPUT_FOLDER, label, f"IPS_{filename}")

        # Lagre augmentert lydfil
        try:
            write(augmented_wav_file, augmented_signal, sample_rate)
            print(f"File successfully saved augmented file {augmented_wav_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
