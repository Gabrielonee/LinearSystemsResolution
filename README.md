# Mini Libreria per la Risoluzione di Sistemi Lineari

Questo progetto implementa una mini-libreria in **Python** per la risoluzione di sistemi lineari con matrici **simmetriche e definite positive**, utilizzando metodi iterativi. È stato sviluppato nell'ambito del corso di Metodi del Calcolo Scientifico (AA 2024/25).

---

## 🚀 Funzionalità

- Implementazione dei seguenti metodi:
  - Metodo di **Jacobi**
  - Metodo di **Gauss-Seidel**
  - Metodo del **Gradiente**
  - Metodo del **Gradiente Coniugato**
- Profilazione automatica (tempo di esecuzione e memoria)
- Calcolo dell'errore relativo rispetto alla soluzione esatta
- Analisi delle performance al variare della tolleranza e del numero massimo di iterazioni
- Supporto al formato **Matrix Market (.mtx)** per matrici sparse

---

## ❓ Come usare il main script

Lo script principale consente di risolvere sistemi lineari sparsi caricando file matriciali .mtx da una cartella o inserendo manualmente una matrice e un vettore di destra attraverso la riga di comando.

Argomenti della riga di comando
  -m o --mode: Richiesto
Scegliere la modalità di input:
	- "file": leggere tutti i file .mtx da una cartella
	- "manual": immissione manuale della matrice e del vettore laterale destro tramite la riga di comando.
	-d o --data-dir: Opzionale, default=data
Percorso della directory contenente i file .mtx (usato solo in modalità file).


### Esempi di utilizzo

1. Risolvere matrici da file .mtx in una cartella

python main.py -m file -d path/to/matrix_folder

- Lo script analizza la cartella per trovare tutti i file .mtx, carica ciascuno di essi come matrice rada ed esegue i risolutori con tolleranze predefinite.
- I risultati (grafici e file JSON) vengono salvati nelle directory output/plots e output/results_json (sottocartelle per i casi con/senza RHS forniti dall'utente).
- Il percorso della cartella è predefinito a data se -d non è specificato.

2. Inserimento manuale della matrice e del vettore del lato destro

python main.py -m manuale

- Il programma chiede di inserire le dimensioni della matrice.
- Quindi si inseriscono i valori di ogni cella riga per riga.
- Infine, si inseriscono i valori del vettore del lato destro.
- Il risolutore viene eseguito su questi input e i risultati vengono salvati in modo simile.


### Output
- Grafici: confronto visivo delle prestazioni dei solutori, salvato in output/plots o output/plots_given
- File JSON: risultati numerici dettagliati salvati in output/results_json o output/results_json_given


---

## 📊 Analisi dei Risultati

Per ogni metodo, vengono confrontate:
- Tolleranza vs Errore
- Iterazioni massime vs Tempo
- Metodo vs Memoria
- Numero di iterazioni vs Tolleranza
---

## 📌 Requisiti
- Python 3.9+
- Compatibilità con sistemi operativi Linux, macOS, Windows

---

## 📚 Riferimenti
- Le matrici di test sono nel formato Matrix Market: [math.nist.gov/MatrixMarket](https://math.nist.gov/MatrixMarket/formats.html)
- Standard di stile seguiti: [PEP8](https://peps.python.org/pep-0008/), [PEP257](https://peps.python.org/pep-0257/)

---

## 👨‍💻 Autori
Francesco Romeo  
Gabriele Soranno
