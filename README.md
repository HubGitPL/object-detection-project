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


**How crosshair movement works**:

- Every other frame is used for aiming to keep FPS high.
- Boxes are sorted by distance to the screen center; the closest is picked.
- Head aim point is 10% from the top of the box; cursor offsets are computed to that point.
- Movement is vector-based with a max step; closer targets shrink the max step for smooth deceleration.
- Tiny offsets (<4 px) are ignored to avoid jitter; large jumps (>40 px shift) retarget to the new head position, otherwise keep interpolating toward the previous point.

**Mouse movement calculation using trigonometry**:

The smooth mouse movement is calculated using trigonometric functions to ensure consistent and natural cursor motion:

1. **Offset calculation**: First, we compute the X and Y offset between the target position (enemy head) and screen center
2. **Distance calculation**: Euclidean distance is calculated using the Pythagorean theorem: `distance = sqrt(offset_x² + offset_y²)`
3. **Dynamic speed adjustment**: Maximum movement per frame is dynamically reduced based on distance to target, creating smooth deceleration as the cursor approaches the target
4. **Direction angle**: We calculate the movement angle using `atan2(offset_y, offset_x)`, which gives us the precise direction to the target
5. **Vector decomposition**: The final movement is decomposed into X and Y components using:
   - `move_x = cos(angle) × movement_distance`
   - `move_y = sin(angle) × movement_distance`

**Why trigonometry?**

Using trigonometric functions ensures that:
- Mouse moves at **constant speed** regardless of direction (diagonal movement isn't faster than horizontal/vertical)
- Movement is **perfectly smooth** and natural-looking in all directions
- The direction vector is **properly normalized**, preventing the common issue where diagonal movement would be √2 times faster
- The algorithm can handle **any angle** with equal precision, making targeting predictable and accurate

**Requirements**:

- Model weights file named `best.pt` in project root
- 1920x1080 screen resolution
- Administrative privileges may be required for mouse control
- **X11 Desktop Environment** - The project was designed for X11 desktop environment due to the screenshot tool used (MSS library)

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
3. **System Auto-Aim** - System przechwytywania ekranu i automatycznego celowania dla gier FPS (CS2)

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


**Jak działa ruch celownika**:

- Do celowania używana jest co druga klatka, aby utrzymać wysokie FPS.
- Ramki są sortowane według odległości od środka ekranu; wybierana jest najbliższa.
- Punkt celowania to głowa, 10% od góry ramki; obliczany jest offset kursora do tego punktu.
- Ruch jest wektorowy z maksymalnym krokiem; dla bliższych celów krok maleje, co zapewnia płynne wyhamowanie.
- Bardzo małe przesunięcia (<4 px) są ignorowane, aby uniknąć drgań; duże skoki (>40 px) wymuszają przeskok do nowej głowy, inaczej kontynuowana jest interpolacja do poprzedniego punktu.

**Obliczanie ruchu myszką z użyciem funkcji trygonometrycznych**:

Płynny ruch myszy jest wyliczany z wykorzystaniem funkcji trygonometrycznych, co zapewnia spójny i naturalny ruch kursora:

1. **Obliczanie przesunięcia**: Najpierw obliczamy przesunięcie X i Y między pozycją docelową (głową przeciwnika) a środkiem ekranu
2. **Obliczanie odległości**: Odległość euklidesowa jest wyliczana za pomocą twierdzenia Pitagorasa: `odległość = sqrt(offset_x² + offset_y²)`
3. **Dynamiczne dostosowanie prędkości**: Maksymalny ruch na klatkę jest dynamicznie redukowany w zależności od odległości do celu, tworząc płynne wyhamowanie w miarę zbliżania się kursora do celu
4. **Kąt kierunku**: Obliczamy kąt ruchu używając `atan2(offset_y, offset_x)`, co daje nam precyzyjny kierunek do celu
5. **Rozkład wektora**: Końcowy ruch jest rozkładany na składowe X i Y za pomocą:
   - `ruch_x = cos(kąt) × dystans_ruchu`
   - `ruch_y = sin(kąt) × dystans_ruchu`

**Dlaczego funkcje trygonometryczne?**

Użycie funkcji trygonometrycznych zapewnia, że:
- Mysz porusza się ze stałą prędkością niezależnie od kierunku (ruch po przekątnej nie jest szybszy niż poziomy/pionowy)
- Wektor kierunku jest prawidłowo znormalizowany, co zapobiega powszechnemu problemowi, gdzie ruch po przekątnej byłby szybszy

**Wymagania**:

- Plik wag modelu o nazwie `best.pt` w katalogu głównym projektu
- Rozdzielczość ekranu 1920x1080
- Uprawnienia administratora mogą być wymagane do sterowania myszą
- **Środowisko X11** - Projekt został przygotowany dla środowiska graficznego X11 ze względu na wykorzystane narzędzie do zrzutów ekranu (biblioteka MSS)

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
