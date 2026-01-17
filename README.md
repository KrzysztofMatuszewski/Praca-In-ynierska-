# Anomaly Detection System

System detekcji anomalii dla NIDS (Network Intrusion Detection System) i HIDS (Host Intrusion Detection System) wykorzystujący autoenkodery.

## Struktura projektu

```
anomalies_detection/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── config/
│   │   ├── data/     # Katalogi na dane i modele
│   │   ├── retrain/
│   │   └── scripts/
│   ├── requirements.txt
│   └── .env.example
└── frontend/         # React + Vite frontend
    ├── src/
    └── package.json
```

## Instalacja

### Backend

1. Przejdź do katalogu backend:
```bash
cd anomalies_detection/backend
```

2. Utwórz środowisko wirtualne Python:
```bash
python3 -m venv .
```

3. Aktywuj środowisko:
```bash
source bin/activate
```

4. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

5. Skopiuj i skonfiguruj plik środowiskowy:
```bash
cp .env.example .env
# Edytuj .env i ustaw swoje wartości dla OpenSearch/ElasticSearch
```

6. Uruchom serwer:
```bash
fastapi dev app/main.py --host 0.0.0.0 --port 8012
```

### Frontend

1. Przejdź do katalogu frontend:
```bash
cd anomalies_detection/frontend
```

2. Zainstaluj zależności:
```bash
npm install
# lub
pnpm install
```

3. Uruchom serwer deweloperski:
```bash
npm run dev
# lub
pnpm dev
```

## Konfiguracja

### Backend (.env)

Plik `.env` zawiera konfigurację połączeń do:
- **HIDS**: Wazuh alerts (OpenSearch)
- **NIDS**: Arkime sessions (ElasticSearch)

Przykładowa konfiguracja znajduje się w `.env.example`.

### Frontend

Frontend domyślnie łączy się z backendem na `http://localhost:8012`.

### Baza danych OpeanSearch

Na potrzeby testu oprogramowania należy podłączyć swój własny OpeanSearch oraz załadować bądź zebrać dane.

Przykłądowy zbiór testowy:
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

## Wymagania

- Python 3.11+
- Node.js 18+
- OpenSearch/ElasticSearch z danymi NIDS/HIDS

## Licencja

MIT
