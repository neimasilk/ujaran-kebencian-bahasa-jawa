# Setup Environment Proyek

## Conda Environment

* Nama Environment: `ujaran`
* Python Version: 3.x (sesuai dengan conda base)
* Package Manager: Anaconda

## Library Dasar
* pandas
* numpy
* jupyterlab
* scikit-learn

## Cara Mengaktifkan Environment
```bash
conda activate ujaran
```

## Cara Export Dependencies
```bash
conda list -e > requirements.txt
```

## Cara Install Dependencies
```bash
conda create -n ujaran python=3.x
conda activate ujaran
conda install pandas numpy jupyterlab scikit-learn
```

---
**Catatan:** File ini berisi informasi teknis tentang setup environment proyek. Gunakan sebagai referensi untuk setup ulang atau dokumentasi. 