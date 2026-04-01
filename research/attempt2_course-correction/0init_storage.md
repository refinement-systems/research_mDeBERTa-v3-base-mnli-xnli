Result files in this project are disorganized. I propose using a SQLite database which will hold:

Dataset (same table structure as the .tsv data sets, with a primary key column, and a name taken from the filename):
- primary key
- `id`
- `name`
- `label`
- `premise` 
- `hypothesis`
- `dataset`
- `config`
- `split`
- `row_idx`

Quantization:
- primary key
- name
- program (relative to `tools` directory) to recreate by appending `--backend=$BACKEND --src=$SOURCE --dest=$DESTINATION` to the command line

Pre-initialize Quantization with a name="reference" and a command=""

Backend:
- primary key
- name

Pre-initialize Backend to one "CPU" and one "CoreML" 

Evaluation: (one per data row x quantization x backend)
- primary key
- Dataset:key
- Quantization:key
- Backend:key
- logit for entailment
- logit for neutral
- logit for contradiction

Programs:
1. Downloader - downloads the data sets and the original model into a given directory
2. Initializer - creates a database, adds the pre-initialization entries described above, loads the datasets into the database
3. Runner - evaluates a given quantization on a given backend on a given data set. Details below

The runner shall:
1. Open the database
2. Select rows from Dataset (by the name given on the command line) that do not have a corresponding Evaluation for the given Backend
3. One by one, evaluate the premise - hypothesis pairs. Insert the result into Evaluation when its done.

Numeric evaluations will be computed from the Evaluation table, by comparing the logits of a given quantization, to the logits from the reference model