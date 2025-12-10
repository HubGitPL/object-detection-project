# Object Detection Project / Projekt Detekcji Obiektów

[English](#english) | [Polski](#polski)

---

## English

### Project Overview

This project implements **real-time object detection in video data streams** for the university subject: **Multimedia Services and Applications**. The solution demonstrates practical applications of computer vision and machine learning for detecting and tracking objects in live video feeds and gaming scenarios.

### Scenario Description

The project focuses on three main scenarios:

1. **Model Training** - Custom training of YOLOv12 models on specialized datasets
2. **Object Detection & Prediction** - Real-time detection on images, videos, and directories
3. **Auto-Aim System** - Advanced real-time screen capture and automatic targeting system for FPS games (CS2)

The auto-aim scenario captures screen content in real-time, detects player models (CT/T classes), and automatically moves the mouse cursor to target enemy heads with smooth interpolation and prediction algorithms.

### Tools and Technologies

#### Core Technologies

- **YOLOv12** - State-of-the-art object detection model (ultralytics implementation)
- **Python 3.10+** - Primary programming language
- **OpenCV** - Image processing and computer vision
- **PyTorch** - Deep learning framework
- **MSS** - Ultra-fast screen capture library
- **Roboflow** - Dataset management and download

#### Key Libraries

- `ultralytics` - YOLOv12 implementation
- `opencv-python` - Image/video processing
- `pynput` - Mouse control for auto-aim
- `mss` - Screen capture
- `python-dotenv` - Environment variable management
- `numpy` - Numerical computations

#### Development Tools

- **uv** - Fast Python package installer and runner
- **Git** - Version control
- **VS Code** - Development environment

### Installation and Setup

#### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip
- CUDA-compatible GPU (recommended for training)
- Roboflow API key (for training scenario)

#### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/HubGitPL/object-detection-project.git
cd object-detection-project
```

2. **Install dependencies**

Using UV (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (for training)

Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_api_key_here
```

### Running the Solution

#### 1. Training a Model

Train a custom YOLOv12 model on your dataset:

```bash
uv run src/train.py
```

This script:

- Downloads the dataset from Roboflow
- Initializes a YOLOv12 model
- Trains for 20 epochs with image size 1024
- Saves weights and training plots

**Configuration**: Edit `src/train.py` to change:

- Dataset source
- Model architecture (yolov12n/s/m/l/x)
- Training epochs
- Image size
- Patience and other hyperparameters

#### 2. Running Predictions

Detect objects in images, videos, or directories:

```bash
uv run src/predict.py --weights best_s.pt --source path/to/media
```

**Arguments**:

- `--weights` (required): Path to trained model weights
- `--source` (required): Path to image/video file or directory
- `--conf` (optional): Confidence threshold (default: 0.25)
- `--imgsz` (optional): Inference image size (default: 640)
- `--output` (optional): Output directory (default: predictions)

**Examples**:

```bash
# Single image
uv run src/predict.py --weights best_s.pt --source image.jpg

# Video file
uv run src/predict.py --weights best_s.pt --source video.mp4 --conf 0.5

# Directory of images
uv run src/predict.py --weights best_s.pt --source ./test_images/ --output ./results
```

#### 3. Auto-Aim System

Real-time screen capture and automatic targeting:

```bash
uv run src/auto_aim.py
```

**Interactive Setup**:

1. Select target class:

   - `0`: Counter-Terrorists (CT)
   - `1`: Terrorists (T)
   - `2`: Everyone (both teams)

2. The system will:

   - Capture screen at 1920x1080 resolution
   - Detect targets in real-time
   - Automatically move cursor to enemy heads
   - Use smooth interpolation and prediction

3. Stop with `Ctrl+C`

**Features**:

- Frame skipping for performance optimization
- Distance-based target prioritization
- Smooth mouse movement with velocity limiting
- Head position targeting (10% from top of bounding box)
- FPS logging and monitoring

**Requirements**:

- Model weights file named `best.pt` in project root
- 1920x1080 screen resolution
- Administrative privileges may be required for mouse control

### Project Structure

```
object-detection-project/
├── src/
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction/inference script
│   └── auto_aim.py       # Auto-aim system
├── docs/                 # Documentation
├── best_*.pt             # Pre-trained model weights
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```

### Model Weights

The project includes several pre-trained YOLOv12 models:

- `best_n.pt` - Nano (fastest, least accurate)
- `best_s.pt` - Small
- `best_m.pt` - Medium
- `best_l.pt` - Large
- `best_x.pt` - Extra Large (slowest, most accurate)

### Authors

- **Artur Binczyk** - Main Developer
- **Jerzy Szyjut** - Team Lead
- **Mateusz Fydrych** - Project Manager
- **Patryk Welenc** - Main Developer

---

## Polski

### Przegląd Projektu

Projekt implementuje **detekcję obiektów w czasie rzeczywistym w strumieniu danych wideo** dla przedmiotu: **Usługi i Aplikacje Multimedialne**. Rozwiązanie demonstruje praktyczne zastosowania wizji komputerowej i uczenia maszynowego do wykrywania i śledzenia obiektów w strumieniach wideo oraz scenariuszach gamingowych.

### Opis Scenariusza

Projekt koncentruje się na trzech głównych scenariuszach:

1. **Trenowanie Modelu** - Niestandardowe trenowanie modeli YOLOv12 na specjalistycznych zbiorach danych
2. **Detekcja i Predykcja Obiektów** - Wykrywanie w czasie rzeczywistym na obrazach, filmach i katalogach
3. **System Auto-Aim** - Zaawansowany system przechwytywania ekranu i automatycznego celowania dla gier FPS (CS2)

Scenariusz auto-aim przechwytuje zawartość ekranu w czasie rzeczywistym, wykrywa modele graczy (klasy CT/T) i automatycznie przesuwa kursor myszy do celowania w głowy przeciwników z płynną interpolacją i algorytmami predykcji.

### Narzędzia i Technologie

#### Główne Technologie

- **YOLOv12** - Najnowocześniejszy model detekcji obiektów (implementacja ultralytics)
- **Python 3.10+** - Główny język programowania
- **OpenCV** - Przetwarzanie obrazów i wizja komputerowa
- **PyTorch** - Framework uczenia głębokiego
- **MSS** - Ultraszybka biblioteka przechwytywania ekranu
- **Roboflow** - Zarządzanie i pobieranie zbiorów danych

#### Kluczowe Biblioteki

- `ultralytics` - Implementacja YOLOv12
- `opencv-python` - Przetwarzanie obrazów/wideo
- `pynput` - Sterowanie myszą dla auto-aim
- `mss` - Przechwytywanie ekranu
- `python-dotenv` - Zarządzanie zmiennymi środowiskowymi
- `numpy` - Obliczenia numeryczne

#### Narzędzia Deweloperskie

- **uv** - Szybki instalator i runner pakietów Python
- **Git** - Kontrola wersji
- **VS Code** - Środowisko deweloperskie

### Instalacja i Konfiguracja

#### Wymagania Wstępne

- Python 3.10 lub nowszy
- Menedżer pakietów UV (zalecany) lub pip
- GPU kompatybilne z CUDA (zalecane do trenowania)
- Klucz API Roboflow (dla scenariusza trenowania)

#### Kroki Instalacji

1. **Sklonuj repozytorium**

```bash
git clone https://github.com/HubGitPL/object-detection-project.git
cd object-detection-project
```

2. **Zainstaluj zależności**

Używając UV (zalecane):

```bash
uv sync
```

Używając pip:

```bash
pip install -r requirements.txt
```

3. **Skonfiguruj zmienne środowiskowe** (do trenowania)

Utwórz plik `.env` w katalogu głównym projektu:

```
ROBOFLOW_API_KEY=twój_klucz_api
```

### Uruchamianie Rozwiązania

#### 1. Trenowanie Modelu

Wytrenuj niestandardowy model YOLOv12 na swoim zbiorze danych:

```bash
uv run src/train.py
```

Skrypt:

- Pobiera zbiór danych z Roboflow
- Inicjalizuje model YOLOv12
- Trenuje przez 20 epok z rozmiarem obrazu 1024
- Zapisuje wagi i wykresy treningowe

**Konfiguracja**: Edytuj `src/train.py` aby zmienić:

- Źródło zbioru danych
- Architekturę modelu (yolov12n/s/m/l/x)
- Liczbę epok treningowych
- Rozmiar obrazu
- Cierpliwość i inne hiperparametry

#### 2. Uruchamianie Predykcji

Wykrywaj obiekty na obrazach, filmach lub w katalogach:

```bash
uv run src/predict.py --weights best_s.pt --source ścieżka/do/mediów
```

**Argumenty**:

- `--weights` (wymagany): Ścieżka do wag wytrenowanego modelu
- `--source` (wymagany): Ścieżka do pliku obrazu/wideo lub katalogu
- `--conf` (opcjonalny): Próg pewności (domyślnie: 0.25)
- `--imgsz` (opcjonalny): Rozmiar obrazu dla inferencji (domyślnie: 640)
- `--output` (opcjonalny): Katalog wyjściowy (domyślnie: predictions)

**Przykłady**:

```bash
# Pojedynczy obraz
uv run src/predict.py --weights best_s.pt --source obraz.jpg

# Plik wideo
uv run src/predict.py --weights best_s.pt --source wideo.mp4 --conf 0.5

# Katalog z obrazami
uv run src/predict.py --weights best_s.pt --source ./obrazy_testowe/ --output ./wyniki
```

#### 3. System Auto-Aim

Przechwytywanie ekranu i automatyczne celowanie w czasie rzeczywistym:

```bash
uv run src/auto_aim.py
```

**Konfiguracja Interaktywna**:

1. Wybierz klasę celu:

   - `0`: Antyterroryści (CT)
   - `1`: Terroryści (T)
   - `2`: Wszyscy (obie drużyny)

2. System będzie:

   - Przechwytywać ekran w rozdzielczości 1920x1080
   - Wykrywać cele w czasie rzeczywistym
   - Automatycznie przesuwać kursor do głów przeciwników
   - Używać płynnej interpolacji i predykcji

3. Zatrzymaj przez `Ctrl+C`

**Funkcje**:

- Pomijanie klatek dla optymalizacji wydajności
- Priorytetyzacja celów na podstawie odległości
- Płynny ruch myszy z ograniczeniem prędkości
- Celowanie w głowę (10% od góry ramki ograniczającej)
- Logowanie i monitorowanie FPS

**Wymagania**:

- Plik wag modelu o nazwie `best.pt` w katalogu głównym projektu
- Rozdzielczość ekranu 1920x1080
- Uprawnienia administratora mogą być wymagane do sterowania myszą

### Struktura Projektu

```
object-detection-project/
├── src/
│   ├── train.py          # Skrypt trenowania modelu
│   ├── predict.py        # Skrypt predykcji/inferencji
│   └── auto_aim.py       # System auto-aim
├── docs/                 # Dokumentacja
├── best_*.pt             # Wagi wytrenowanych modeli
├── pyproject.toml        # Zależności projektu
└── README.md             # Ten plik
```

### Wagi Modeli

Projekt zawiera kilka wytrenowanych modeli YOLOv12:

- `best_n.pt` - Nano (najszybszy, najmniej dokładny)
- `best_s.pt` - Small (mały)
- `best_m.pt` - Medium (średni)
- `best_l.pt` - Large (duży)
- `best_x.pt` - Extra Large (najwolniejszy, najbardziej dokładny)

### Autorzy

- **Artur Binczyk** - Główny Programista
- **Jerzy Szyjut** - Lider Zespołu
- **Mateusz Fydrych** - Project Manager
- **Patryk Welenc** - Główny Programista
