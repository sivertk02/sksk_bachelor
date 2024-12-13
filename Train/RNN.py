'''BESKRIVELSE
Dette programmet er laget for å trene et kunstig nevralt nettverk for klassifisering av data ved hjelp av en LSTM-basert RNN-modell. 
Programmet behandler JSON-filer med transofmert lyddata som først deles inn i trenings-, validerings-, og testsett.
ChatGPT er blitt brukt til å generere deler av programmet

Kode er basert på:
Github: https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master/19-%20How%20to%20implement%20an%20RNN-LSTM%20for%20music%20genre%20classification/code
Youtube: https://www.youtube.com/watch?v=4nXI0h2sq2I&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=19

Sist oppdatert: 13.12.2024
'''

from typing import Union, List 
import json # Importering av data fra JSON-filer
import numpy as np # Bruk av matriser
import os # For håndtering av filer og filbaner
from tensorflow import keras # Byggnig av modell
import matplotlib.pyplot as plt  # For plotting
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # For bedre visualisering av konfuskjonsmatrise
plt.switch_backend("Agg") # Endrer filer til grafikk i stedet for et vindu (siden jeg bruker WSL2 som ikke støtter GUI)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_INPUT_TRAIN_JSON = os.path.join(_BASE_DIR, 'train_60.json')
_INPUT_VAL_JSON = os.path.join(_BASE_DIR, 'val_20.json')
_INPUT_TEST_JSON = os.path.join(_BASE_DIR, 'test_LoVe.json')

_INPUT_TRAIN_PS_1_JSON = os.path.join(_BASE_DIR, 'train_60_PS_seed_1.json')
_INPUT_TRAIN_IPS_1_JSON = os.path.join(_BASE_DIR, 'train_60_IPS_seed_1.json')

_INPUT_TRAIN_LIST_JSON = [_INPUT_TRAIN_JSON, _INPUT_TRAIN_IPS_1_JSON]

_OUTPUT_DIR = os.path.join(_BASE_DIR, 'RNN_LoVe_output_1600epochs')

def hent_filinfo(filbane: str) -> None:
    """Henter filtype og størrelse, og skriver ut informasjonen."""
    if not os.path.exists(filbane):
        raise FileNotFoundError(f"Filen '{filbane}' finnes ikke.")
    
    # Henter ut filtype
    _, filtype = os.path.splitext(filbane)

    # Hente filstørrelse i byte
    filstorrelse = os.path.getsize(filbane)
    filstorrelse_gb = round(filstorrelse / (1024 ** 3), 4)  # Runder av til 4 desimaler

    print(f'{filbane} eksisterer, har filtype: {filtype}, og størrelse: {filstorrelse_gb} GB')

def sjekk_fil(filbaner: Union[str, List[str]]) -> None:
    """Sjekker om filen(e) finnes, og skriver ut filtypen og størrelsen i GB."""
    # Hvis filbaner er en streng, konverter til liste
    if isinstance(filbaner, str):
        hent_filinfo(filbaner)
    else:
        for filbane in filbaner:
            hent_filinfo(filbane)

def json_til_array(filepaths: Union[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Laster JSON-filer og returnerer x- og y-arrays.
    
    Args:
        filepaths (Union[str, list[str]]): Sti til en enkelt JSON-fil eller en liste av stier.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: x- og y-data som numpy-arrays.
    """
    # Hvis input er en enkelt filbane, konverter til liste for enklere behandling
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    x_data_list = []
    y_data_list = []

    # Laster data fra hver fil
    for filepath in filepaths:
        with open(filepath, 'r') as fp:
            data = json.load(fp)
        
        # Ekstraherer magnitudes og labels fra JSON-innholdet
        x_data_list.extend([item['magnitude'] for item in data['cqt']])
        y_data_list.extend(data['labels'])
    
    # Konverterer lister til numpy-arrays
    x_data = np.array(x_data_list)
    y_data = np.array(y_data_list)

    return x_data, y_data

def bygg_modell(input_shape: tuple[int, int]) -> keras.Sequential:
    '''Genererer RNN-LSTM
    
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    '''

    # Definerer modellens topologi
    model = keras.Sequential()

    # 2 LSTM lag
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    '''Parametere
    64 # Number of units
    return_sequences= # Vi ønsker et sequence to sequence lag, fordi vi ønsker å sende outputen videre
    input_shape= # Dimensjonene til inputvektorene
    '''
    
    # Dense lag
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0,3)) # unngår overfitting med dropout


    # output layer (4 kategorier)
    model.add(keras.layers.Dense(4, activation='softmax'))

    return model

def plot_loss(history: keras.callbacks) -> None:
    """Plotter tap over epoker."""
    plt.figure()  # Start en ny figur
    plt.plot(history.history['loss'], label='Treningstap')
    plt.plot(history.history['val_loss'], label='Valideringstap')
    plt.title('Tap over Epoker')
    plt.xlabel('Epoker')
    plt.ylabel('Tap')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(_OUTPUT_DIR, 'loss_plot.png')) 

def plot_accuracy(history: keras.callbacks) -> None:
    '''Plotter tap over epoker'''
    plt.figure()  # Start en ny figur
    plt.plot(history.history['accuracy'], label='Treningnøyaktighet')
    plt.plot(history.history['val_accuracy'], label='Valideringsnøyaktighet')
    plt.title('Nøyaktighet over Epoker')
    plt.xlabel('Epoker')
    plt.ylabel('Nøyaktighet')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(_OUTPUT_DIR, 'accuracy_plot.png'))

def plot_confusion_matrix(y_true, y_pred):
    '''Plotter konfuskjonsmatrise.'''
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Konfuskjonsmatrise')
    plt.xlabel('Predikerte klasser')
    plt.ylabel('Sanne klasser')
    plt.savefig(os.path.join(_OUTPUT_DIR, 'confusion_matrix.png'))  # Lagre plottet

if __name__ == '__main__':
    # Sjekker filenes gyldighet
    sjekk_fil(_INPUT_TRAIN_LIST_JSON)
    sjekk_fil(_INPUT_VAL_JSON)
    sjekk_fil(_INPUT_TEST_JSON)

    x_train, y_train = json_til_array(_INPUT_TRAIN_LIST_JSON)
    x_val, y_val = json_til_array(_INPUT_VAL_JSON)
    x_test, y_test = json_til_array(_INPUT_TEST_JSON)

    # Bygger RNN-nettverk
    input_shape = (x_train.shape[1], x_train.shape[2]) # 48 x 64 
    model = bygg_modell(input_shape)

    # Kompilerer nettverker
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=['accuracy'])

    model.summary() # Skriver ut en oppsummering

    # Trener opp RNN og lagrer opptreningsloggen for modellering 
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=1600)
    plot_loss(history)
    plot_accuracy(history)

    # Evaluerer modellen og skriver ut informasjon
    loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
    print(f'Testsett - Loss: {loss}. - Accuracy: {accuracy}')

    # Klassifiseringsrapport og confusions-matrise
    y_pred = model.predict(x_test, batch_size=32, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, zero_division=0))
    plot_confusion_matrix(y_test, y_pred_classes)

