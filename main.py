import os

from utilities.common import MatrixReader
from solver import solver_matrix

FOLDER_PATH = "/Users/fraromeo/Documents/02_Areas/\
University/LM/LM_24-25/SEM2/MdCS/dati"

# Directory che contiene i file .mtx
directory = os.fsencode(FOLDER_PATH)
# Iterazione su tutti i file della directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # Considera solo i file con estensione .mtx
    if filename.endswith(".mtx"):
        # Costruzione del path completo
        matrixPath = os.path.join(os.fsdecode(directory), filename)

        # Caricamento della matrice sparsa dal file .mtx
        A_sparse = MatrixReader(matrixPath)
        responses = solver_matrix(A_sparse)

        for response in responses:
            print(response)
