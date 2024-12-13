'''BESKRIVELSE
Dette programmet er laget for å trene et kunstig nevralt nettverk for klassifisering av data ved hjelp av et CNN . 
Programmet behandler JSON-filer med transofmert lyddata som først deles inn i trenings-, validerings-, og testsett.
Chat GPT er blitt brukt til å generere deler av programmet

Kode er basert på:
Github: https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master/16-%20How%20to%20implement%20a%20CNN%20for%20music%20genre%20classification/code
Youtube: https://www.youtube.com/watch?v=dOG-HxpbMSw&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=16

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
_INPUT_TEST_JSON = os.path.join(_BASE_DIR, 'test_20.json')


_INPUT_TRAIN_LIST_JSON = [_INPUT_TRAIN_JSON]

_OUTPUT_DIR = os.path.join(_BASE_DIR, 'CNN_output')

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

def build_model(input_shape):
    # Lager modellen
    model = keras.Sequential()  # Modellen legger til lagene sekvensielt i arkitekturen

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    '''
    Kernal = 32: antall filtre
    Gridsize = 3x3: Størrelse på filtre
    Activation = relu: aktiveringsfunksjon
    '''
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same')) # Legger til max-pooling
    model.add(keras.layers.BatchNormalization()) # En prosess som normaliserer aktiveringen (gjør opptreningen raskere)

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # Minker størrelse på filter
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same')) 
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape)) # Minker størrelse på filter
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2,2), padding='same')) 
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten()) # Vi gjør om ouputen fra 2-dimensjonell array til en 1-dimensjonell array
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))) 
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) # Legger til dropout for å unngå overfitting  
    
    # output layer (bruker softmax)
    model.add(keras.layers.Dense(4, activation='softmax')) # Vi må ha like mange nevroner som kategorier

    # Returnerner modellen
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

if __name__ == "__main__":
    # Lager trenings, validering og test-set
    x_train, y_train = json_til_array(_INPUT_TRAIN_LIST_JSON)
    x_val, y_val = json_til_array(_INPUT_VAL_JSON)
    x_test, y_test = json_til_array(_INPUT_TEST_JSON)

    # Tensorflow forventer en 3d array for hver sample for konvolusjon
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Bygger CNN-nettet
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)

    # Kompilerer nettverker
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=['accuracy'])

    model.summary() # Skriver ut en oppsummering

    # Trener opp CNN og lagrer opptreningsloggen for modellering 
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50)
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

