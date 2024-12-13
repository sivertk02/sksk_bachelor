'''BESKRIVELSE
Dette programmet konverterer lydfilene til CQT, og lagrer det i JSON-format.
Kode er inspirert av video: https://www.youtube.com/watch?v=Oa_d-zaUti8&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=11
med tilhørende github: https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master/11-%20Preprocessing%20audio%20data%20for%20deep%20learning/code

Koden er blitt lastet opp og endret av ChatGPT til å konvertere til CQT
Valg av konstantene er inspirert av Irfran et al. i opgaven:
https://download.ssrn.com/apac/258ce311-10ac-436e-8b5c-b1793c03984a-meca.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCrvH4YVGE5%2FON7iJBufbLv3yCfE8TMnWJ4wjxEdmXOUAIhAKlBuSOcM7wT3jXpoEMruyjWV8mP1vx5NRmPfycaS%2B2EKr0FCHQQBBoMMzA4NDc1MzAxMjU3IgxqDaty2yp4uCSl%2FxYqmgUHE85k16eJXfcQqRc%2F5luhKU74%2Bi0Wi%2BsLPOZRO9lTtvVLxp0ZgbTXSiGUlxQgY3Gw5UjZuwwyUuNoi8Cx2HP9UpeYFLslI4JAOfqdokQUbyEzsic66Fu05vBf%2FhFH7WQiVIoiHVpj8lUPZI%2Fl2oP%2FIct7dPASgI6k%2FYtL8TbtS7WGQRXMzitTfEP0EKuBeq2sTCtdOnFomKVbDHMBSgiIbSWF8n8RRhIH5bfZO0Ne%2BdSwPDKysu5cvBu6fY6aMVvz%2FsRmVUPwsgOblpmCVo42gKXcHq74%2Fo3YKc2FkgDEtIeAcBb%2ByAm4EjVac3u1DhA4ywn2IAP7yRfGFL6HtG9dm0Z9oKd2jPskPZpHM6lV%2FuB2N0Z0cLbK7TNkro31Z4OtjNPsj%2F6rrwT5Z9Oo0EpJpp2ZXsY9VJgv6z5nSeLUWVh5FfVK38L%2BMeTWIz7vUNxQrTImKe4HgGm8e5RPk88nQzWLXgrjeaAUtUFwrFpFzaXb4ByWfdGjEHUu2Zvntw4%2BoSnU%2BnYBJiWsqZZ4opl7%2BhCQ78d78w0pkO9R%2FPGafLuWH68oBBFsmlGFzm8jelcWpHxiw9%2FEq8asJz0Evo9kQg8%2FSV%2F6nQsL7ND6i5mfZ%2BCEqob76cCw5qXfBSAQQAHqcPGdDdmbboqGZlU30OZsOhuH0xlFGskpz7%2FzrIXy1D0paSQeyvDJHcgsh41sTxbl69U1AHBoqjVgY1bY6zmNXo9UBkntbHQsUb%2BhLS1dcXmtAVJltuKniwwwO2hrlUrg1DtEUhwZWnAahSBVUACaycfNtu9TkgURTkMDDnFlo01X0eG88w6d3IgsD4%2FT3RbRkwUudj8lIoi2WJi2uS%2F7haRKClVwR%2FRLHDtTOJzVvmrCg9I9YytpDbEw9bamugY6sAGsgdlpBqkmMXxRm2QcxCjzCLeQj6nMEGWvFxJpvP1Kka4AyOJCX5ysx7UDjH9GMuUhAa26DGm3atLJZ0eEa%2BJew0aU7fgnJddW5dL1RbB7h3GaHH0TmTAhErwbOoO61wDYlDLjL3e8oeBKxdnycCV%2F3d9yeSTf%2FsUVSauEHVF1%2BfKyw9uyVZz4OZ2ApWQVKBq0ZQQsL34nugPVy1MZWhcoV%2F5Ii8m2BWMieOf9MYczYg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241129T115132Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWEWYH5DSBZ%2F20241129%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=9782467a5c9daa7fc567a54ee6821caabad93b45af629d628d57aa405c81203d&abstractId=4347042
slik at resultatene kan sammenliknes

Sist oppdatert: 13.12.2024
'''

import os # Verifisering av filbaner
from numpy import abs
import librosa # Behandling av lydfiler
import math
import json # Konvertering til JSON-format

DATASET_PATH = '' 
JSON_PATH = '.json' # Må legge til filtype .json

SAMPLE_RATE = 16000 # 16kHz (samme som M. Irfran et al.) 
DURATION = 3 # Hvert input-vektor 1 x n varer i 3 sekunder
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_cqt(dataset_path: str, json_path: str, n_bins: int=64, hop_length: int=1012, num_segments:int=1) -> None:
    '''VALG AV VERDIER: Verdiene er valgt slik at dimensjonene for segment  blir 68 x 48 
    n_bins : int -> anntall frekvensbånd som blir analysert 
    hop_legnth : float -> hvor ofte CQT blir lagt 
    num_segments : float -> Dersom vi har større lydfiler kan vi dele dem opp i flere segmenter
    '''

    data = {
        "mapping": [], #
        "cqt": [], #
        "labels": [] #
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments) # 1 
    expected_num_cqt_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            
            print("\nProcessing {}".format(semantic_label))
            
            for f in filenames: 
                file_path = os.path.join(dirpath, f)
                
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
                    cqt = librosa.cqt(y=signal[start_sample:finish_sample],  sr=sr, n_bins=n_bins,hop_length=hop_length)
                    
                    cqt = cqt.T
                    
                    if len(cqt) == expected_num_cqt_vectors_per_segment:

                        # Beregn amplituder og fase
                        cqt_magnitude = abs(cqt).tolist()  # Absoluttverdi
                        #cqt_phase = np.angle(cqt).tolist()  # Faseinformasjon
                        
                        # Lagrer både magnitude og fase i en ordbok
                        data["cqt"].append({
                            "magnitude": cqt_magnitude,
                        #    "phase": cqt_phase
                        })

                        data["labels"].append(i-1)
                        print("{}, segment: {}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    if not os.path.isdir(DATASET_PATH):
        print(f"Directory not found: {DATASET_PATH}")
    else:
        save_cqt(DATASET_PATH, JSON_PATH, num_segments=1) # Gjør hele lydfiler om til et segement
        print("Program finished sucessfully")
