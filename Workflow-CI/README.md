# Workflow CI for Wine Quality Prediction
Repositori ini berisi implementasi Workflow CI (Continuous Integration) untuk pelatihan ulang model machine learning menggunakan MLflow Project, GitHub Actions, dan DagsHub.

## Struktur Folder
Workflow-CI
├── .github
│ └── workflows
│ └── ci.yml # Workflow GitHub Actions
├── MLProject
│ ├── modelling.py # Script pelatihan model ML
│ ├── conda.yaml # Dependency environment untuk MLflow Project
│ ├── MLproject # Konfigurasi MLflow Project
│ ├── winequality_preprocessed.csv # Dataset preprocessed
├── DagsHub.txt # Informasi koneksi ke DagsHub
├── README.md


## DagsHub Tracking Info
MLflow Tracking URI: https://dagshub.com/TheDuke03/Eksperimen_SML_Muhammad-Firdaus.mlflow
Experiment Name: Eksperimen_Wine_DagsHub
Run Name: casual-stag-950


## Cara Kerja
- Ketika kode didorong ke repositori ini (trigger), GitHub Actions akan menjalankan MLflow Project.
- Script `modelling.py` akan dilatih ulang menggunakan data preprocessed.
- Artefak seperti model, metrik, dan visualisasi akan dikirim ke MLflow Tracking (DagsHub).

## Requirement
- Akun DagsHub
- Token akses di-repositori GitHub (sebagai secret)
- Docker Hub (jika ingin sampai level advanced)

## Author
- Muhammad Firdaus


