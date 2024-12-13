'''BEKSRIVELSE 
Dette programmet splitter datasettet inn i trenings- validerings og testsett
Denne koden er generert ved bruk av ChatGPT

Sist oppdatert: 13.12.2024
'''
import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_PATH = ''
OUTPUT_PATH = ''

# Funksjon for å samle alle .wav-filer i datasettet og finne kategoriene automatisk
def get_wav_files_and_labels(dataset_path):
    files = []
    labels = []
    categories = next(os.walk(dataset_path))[1]  # Henter mappenavnene som representerer kategorier
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                files.append(os.path.join(category_path, file))
                labels.append(category)  # Label er kategorien
    return files, labels, categories

# Funksjon for å kopiere filer til de respektive mapper
def copy_files(file_list, label_list, dest_dir):
    for file_path, label in zip(file_list, label_list):
        # Bestem destinasjonsmappen for kategorien (label)
        category_dir = os.path.join(dest_dir, label)
        
        # Opprett mappen hvis den ikke finnes
        os.makedirs(category_dir, exist_ok=True)
        
        # Full filbane for destinasjonsfilen
        dest_path = os.path.join(category_dir, os.path.basename(file_path))
        
        # Kopier filen til destinasjonsmappen
        shutil.copy(file_path, dest_path)

# Hovedlogikk for splitting
if __name__ == "__main__":
    # Hent alle filer og deres labels, samt kategoriene
    wav_files, labels, categories = get_wav_files_and_labels(DATASET_PATH)
    
    # Split datasettet i trenings-, validerings- og testsett (60% train, 20% validering, 20% test)
    X_train, X_temp, y_train, y_temp = train_test_split(wav_files, labels, test_size=0.4, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Opprett output-mapper for trenings-, validerings- og testsett
    train_dir = os.path.join(OUTPUT_PATH, 'train')
    val_dir = os.path.join(OUTPUT_PATH, 'val')
    test_dir = os.path.join(OUTPUT_PATH, 'test')

    # Kopier filer til treningsmappen
    print("Kopierer treningsfiler...")
    copy_files(X_train, y_train, train_dir)
    
    # Kopier filer til valideringsmappen
    print("Kopierer valideringsfiler...")
    copy_files(X_val, y_val, val_dir)
    
    # Kopier filer til testmappen
    print("Kopierer testfiler...")
    copy_files(X_test, y_test, test_dir)
    
    print("Ferdig med å splitte og kopiere datasettet.")

