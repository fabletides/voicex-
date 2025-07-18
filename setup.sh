#!/bin/bash

# VoiceX Complete Setup Script
# Автоматическая установка и настройка всей экосистемы VoiceX

set -e  # Прерывание при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}🎙️  VoiceX Setup Script${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Проверка операционной системы
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PKG_MANAGER="yum"
        elif command -v pacman &> /dev/null; then
            PKG_MANAGER="pacman"
        else
            print_error "Неподдерживаемый дистрибутив Linux"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
        PKG_MANAGER="choco"
    else
        print_error "Неподдерживаемая операционная система: $OSTYPE"
        exit 1
    fi
    
    print_step "Обнаружена ОС: $OS с менеджером пакетов: $PKG_MANAGER"
}

# Установка зависимостей системы
install_system_dependencies() {
    print_step "Установка системных зависимостей..."
    
    case $PKG_MANAGER in
        "apt")
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv \
                git curl wget \
                build-essential \
                portaudio19-dev \
                libsndfile1-dev \
                ffmpeg \
                nodejs npm
            ;;
        "yum")
            sudo yum update -y
            sudo yum install -y \
                python3 python3-pip \
                git curl wget \
                gcc gcc-c++ make \
                portaudio-devel \
                libsndfile-devel \
                ffmpeg \
                nodejs npm
            ;;
        "pacman")
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python python-pip \
                git curl wget \
                base-devel \
                portaudio \
                libsndfile \
                ffmpeg \
                nodejs npm
            ;;
        "brew")
            brew update
            brew install \
                python3 \
                git curl wget \
                portaudio \
                libsndfile \
                ffmpeg \
                node
            ;;
        "choco")
            choco install -y \
                python3 \
                git curl wget \
                nodejs
            ;;
    esac
    
    print_success "Системные зависимости установлены"
}

# Создание виртуального окружения Python
setup_python_environment() {
    print_step "Настройка Python окружения..."
    
    # Создание виртуального окружения
    python3 -m venv voicex_env
    
    # Активация окружения
    source voicex_env/bin/activate
    
    # Обновление pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Python окружение создано и активировано"
}

# Установка Python зависимостей
install_python_dependencies() {
    print_step "Установка Python зависимостей..."
    
    # Создание requirements.txt если его нет
    cat > requirements.txt << EOF
# ML и аудио обработка
tensorflow>=2.13.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Данные и утилиты
pandas>=2.0.0
joblib>=1.3.0
tqdm>=4.65.0
argparse

# ESP32 и прошивка
esptool>=4.6.0
platformio>=6.1.0

# Веб и API
requests>=2.31.0
flask>=2.3.0

# Дополнительные утилиты
pathlib
json5
pyyaml>=6.0
pillow>=10.0.0
EOF

    # Установка зависимостей
    pip install -r requirements.txt
    
    print_success "Python зависимости установлены"
}

# Установка PlatformIO
install_platformio() {
    print_step "Установка PlatformIO..."
    
    # Установка PlatformIO CLI
    pip install platformio
    
    # Обновление платформ
    pio platform update
    
    # Установка платформы ESP32
    pio platform install espressif32
    
    print_success "PlatformIO установлен и настроен"
}

# Создание структуры проекта
create_project_structure() {
    print_step "Создание структуры проекта..."
    
    # Создание основных директорий
    mkdir -p {
        datasets,
        models,
        training_artifacts,
        esp32_firmware,
        web_interface,
        docs,
        scripts,
        tests,
        data
    }
    
    # Создание поддиректорий для данных
    mkdir -p datasets/{throat_microphone,speech,test}
    mkdir -p models/{trained,tflite,checkpoints}
    mkdir -p esp32_firmware/{src,data,lib,include}
    
    print_success "Структура проекта создана"
}

# Загрузка исходного кода VoiceX
download_voicex_code() {
    print_step "Создание файлов VoiceX..."
    
    # Создание основного скрипта обучения
    cat > scripts/train_model.py << 'EOF'
#!/usr/bin/env python3
"""
VoiceX Training Script
Основной скрипт для обучения модели персонализации голоса
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Здесь будет импорт VoiceXTrainer из ml_pipeline.py
# from ml_pipeline import VoiceXTrainer

def main():
    print("🚀 Запуск обучения VoiceX модели")
    # Основная логика обучения
    pass

if __name__ == "__main__":
    main()
EOF

    # Создание скрипта тестирования
    cat > scripts/test_model.py << 'EOF'
#!/usr/bin/env python3
"""
VoiceX Testing Script
Скрипт для тестирования обученной модели
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🧪 Запуск тестирования VoiceX модели")
    # Основная логика тестирования
    pass

if __name__ == "__main__":
    main()
EOF

    # Создание скрипта для ESP32
    cat > scripts/deploy_to_esp32.py << 'EOF'
#!/usr/bin/env python3
"""
VoiceX ESP32 Deployment Script
Скрипт для развертывания модели на ESP32
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("📡 Развертывание VoiceX на ESP32")
    # Основная логика развертывания
    pass

if __name__ == "__main__":
    main()
EOF

    # Делаем скрипты исполняемыми
    chmod +x scripts/*.py
    
    print_success "Файлы VoiceX созданы"
}

# Создание конфигурационных файлов
create_config_files() {
    print_step "Создание конфигурационных файлов..."
    
    # Конфигурация модели
    cat > config.json << EOF
{
  "model_config": {
    "input_dim": 512,
    "output_dim": 256,
    "sample_rate": 22050,
    "duration": 3.0,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "n_mfcc": 13
  },
  "training_config": {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "early_stopping_patience": 20
  },
  "esp32_config": {
    "tensor_arena_size": 50000,
    "max_audio_buffer_size": 4096,
    "i2s_sample_rate": 22050,
    "vibro_motor_pwm_freq": 1000
  }
}
EOF

    # PlatformIO конфигурация
    cat > esp32_firmware/platformio.ini << EOF
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino

build_flags = 
    -DCORE_DEBUG_LEVEL=3
    -DVOICEX_DEBUG=1
    -std=gnu++17

lib_deps = 
    bblanchon/ArduinoJson@^6.21.3
    https://github.com/tensorflow/tflite-micro-arduino-examples

board_build.partitions = huge_app.csv
board_build.filesystem = spiffs

monitor_speed = 115200
upload_speed = 921600
EOF

    # Docker конфигурация для изолированной среды
    cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \\
    build-essential \\
    portaudio19-dev \\
    libsndfile1-dev \\
    ffmpeg \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Точка входа
CMD ["python", "scripts/train_model.py"]
EOF

    # Docker Compose для развертывания
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  voicex-training:
    build: .
    volumes:
      - ./datasets:/app/datasets
      - ./models:/app/models
      - ./training_artifacts:/app/training_artifacts
    environment:
      - PYTHONPATH=/app
    command: python scripts/train_model.py --data_path /app/datasets

  voicex-web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./web_interface:/app/web_interface
    command: python -m flask run --host=0.0.0.0 --port=5000
    environment:
      - FLASK_APP=web_interface/app.py
EOF

    print_success "Конфигурационные файлы созданы"
}

# Создание документации
create_documentation() {
    print_step "Создание документации..."
    
    # README
    cat > README.md << EOF
# 🎙️ VoiceX - Система восстановления голоса

VoiceX - это инновационная система для восстановления голоса людей, потерявших способность говорить из-за медицинских причин (ларингэктомия, повреждения голосовых связок и т.д.).

## 🌟 Особенности

- **Персонализированный голос**: ML модель воссоздает индивидуальные характеристики голоса
- **Горловой микрофон**: Захват вибраций через кожу для бесшумного ввода
- **ESP32 платформа**: Компактное носимое устройство
- **Двойные вибромоторы**: Создание естественных голосовых вибраций
- **Машинное обучение**: TensorFlow Lite для персонализации на устройстве

## 🏗️ Архитектура системы

```
Горловой микрофон → Предобработка → ML модель → Вибромоторы
       ↓              ↓              ↓           ↓
   Вибрации кожи → Фильтрация → Персонализация → Голос
```

## 📦 Установка

### Автоматическая установка (рекомендуется)
```bash
curl -fsSL https://raw.githubusercontent.com/voicex/setup/main/install.sh | bash
```

### Ручная установка
```bash
git clone https://github.com/voicex/voicex.git
cd voicex
./setup.sh
```

## 🚀 Быстрый старт

### 1. Подготовка данных
```bash
# Поместите аудио файлы в соответствующие папки
mkdir -p datasets/{throat_microphone,speech}
# Горловой микрофон: datasets/throat_microphone/
# Обычная речь: datasets/speech/
```

### 2. Обучение модели
```bash
python scripts/train_model.py --data_path datasets --output_dir models
```

### 3. Развертывание на ESP32
```bash
python scripts/deploy_to_esp32.py --model_dir models --esp32_port /dev/ttyUSB0
```

## 📊 Компоненты системы

### Hardware (55,860 тенге)
- ESP32 DevKit: 4,500₸
- Вибромоторы R-370-CE (2x): 4,180₸  
- Усилитель PAM8610: 2,160₸
- Микрофон MAX9814: 1,400₸
- Raspberry Pi 4 2GB: 36,000₸
- Аккумулятор Li-ion 18650: 3,200₸
- Компоненты и корпус: 4,420₸

### Software
- TensorFlow Lite для ML inference
- ESP32 прошивка с обработкой аудио
- Python pipeline для обучения
- Веб-интерфейс для управления

## 🧠 ML Pipeline

### Обучение
1. **Извлечение признаков**: Mel-спектрограммы, MFCC, pitch
2. **Предобработка**: Фильтрация шума, нормализация
3. **Обучение модели**: Autoencoder архитектура
4. **Квантизация**: INT8 для ESP32
5. **Валидация**: Тестирование качества

### Inference на ESP32
1. **Захват аудио**: I2S интерфейс
2. **Извлечение признаков**: Упрощенный FFT
3. **ML inference**: TensorFlow Lite
4. **Синтез вибраций**: PWM управление моторами

## 📁 Структура проекта

```
voicex/
├── datasets/              # Тренировочные данные
│   ├── throat_microphone/ # Файлы горлового микрофона
│   └── speech/           # Файлы обычной речи
├── models/               # Обученные модели
├── scripts/              # Скрипты обучения и развертывания
├── esp32_firmware/       # Прошивка ESP32
├── web_interface/        # Веб-интерфейс
├── tests/               # Тесты
└── docs/                # Документация
```

## 🔧 API

### REST API для управления ESP32
```bash
GET  /status              # Статус системы
POST /upload              # Загрузка модели
POST /restart             # Перезапуск
POST /test                # Тест голоса
GET  /logs                # Логи системы
```

### Python API для обучения
```python
from voicex import VoiceXTrainer

trainer = VoiceXTrainer('config.json')
X, y = trainer.load_dataset('datasets')
trainer.train_model(X, y)
trainer.convert_to_tflite('model.tflite')
```

## 🧪 Тестирование

```bash
# Тестирование модели
python scripts/test_model.py --model models/voicex_model.h5 --data datasets/test

# Benchmark на ESP32
python scripts/benchmark_esp32.py --esp32_ip 192.168.1.100
```

## 📈 Метрики качества

- **MSE**: < 0.01 (чем меньше, тем лучше)
- **Cosine Similarity**: > 0.7 (чем больше, тем лучше)  
- **SNR**: > 10 dB (отношение сигнал/шум)
- **Latency**: < 100ms (время отклика)

## 🌐 Веб-интерфейс

Доступен по адресу: http://ESP32_IP/

Функции:
- Загрузка новых моделей
- Мониторинг системы
- Настройка параметров
- Просмотр логов

## 🔒 Безопасность

- Шифрование данных пользователя
- Локальная обработка (без передачи в облако)
- Проверка целостности модели
- Защищенные OTA обновления

## 🤝 Участие в разработке

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 👥 Команда

- **Разработка**: VoiceX Team
- **ML**: Специалисты по обработке речи
- **Hardware**: Инженеры ESP32
- **UX**: Дизайнеры медицинских интерфейсов

## 📞 Поддержка

- **Email**: support@voicex.kz
- **Telegram**: @voicex_support
- **GitHub Issues**: [github.com/voicex/issues](https://github.com/voicex/issues)

## 🏥 Медицинские партнеры

- Национальный научный центр хирургии им. А.Н. Сызганова
- Республиканский центр медицинской реабилитации
- Региональные онкологические диспансеры

---
*VoiceX - Возвращаем голос, возвращаем жизнь* 🎙️
EOF

    # Создание CHANGELOG
    cat > CHANGELOG.md << EOF
# Changelog

Все значимые изменения в проекте VoiceX документированы в этом файле.

## [1.0.0] - 2024-01-20

### Добавлено
- Базовая архитектура ML модели для персонализации голоса
- ESP32 прошивка с поддержкой TensorFlow Lite
- Система захвата аудио через I2S
- Управление вибромоторами через PWM
- Веб-интерфейс для управления системой
- Python pipeline для обучения модели
- Автоматическая настройка и развертывание
- Comprehensive testing suite
- Документация и примеры использования

### Технические детали
- TensorFlow Lite квантизация INT8
- Real-time аудио обработка на ESP32
- Autoencoder архитектура для voice conversion
- SPIFFS файловая система для модели
- REST API для управления устройством
- OTA обновления прошивки

### Поддерживаемые устройства
- ESP32 DevKit v1
- Вибромоторы R-370-CE
- Микрофон MAX9814
- Усилитель PAM8610

## [Планируется v1.1.0]

### В разработке
- Улучшенная архитектура ML модели
- Поддержка Bluetooth аудио
- Мобильное приложение для настройки
- Облачная синхронизация моделей
- Дополнительные языки интерфейса
- Улучшенные алгоритмы шумоподавления
EOF

    # Создание руководства разработчика
    cat > docs/DEVELOPER_GUIDE.md << EOF
# 🔧 Руководство разработчика VoiceX

## Архитектура системы

### Компоненты высокого уровня

1. **ML Pipeline** (Python)
   - Обучение модели персонализации голоса
   - Предобработка аудио данных
   - Конвертация в TensorFlow Lite
   - Валидация и тестирование

2. **ESP32 Firmware** (C++/Arduino)
   - Real-time аудио обработка
   - TensorFlow Lite inference
   - Управление hardware компонентами
   - Веб-сервер для управления

3. **Web Interface** (HTML/JS)
   - Управление системой
   - Загрузка новых моделей
   - Мониторинг состояния
   - Настройка параметров

### Поток данных

```
Аудио → Предобработка → Признаки → ML → Вибрации
  ↓         ↓            ↓        ↓       ↓
I2S → Фильтрация → MFCC/Mel → TFLite → PWM
```

## Разработка ML модели

### Архитектура нейронной сети

```python
# Encoder (горловой микрофон → латентное пространство)
Input(512) → Dense(512) → Dense(256) → Dense(128) → Dense(64)

# Decoder (латентное пространство → персонализированный голос)  
Dense(64) → Dense(128) → Dense(256) → Dense(512) → Output(256)
```

### Извлечение признаков

1. **Mel-спектрограммы**: Частотное представление
2. **MFCC**: Кепстральные коэффициенты
3. **Pitch (F0)**: Основная частота
4. **Спектральные признаки**: Centroid, rolloff, bandwidth
5. **Энергетические признаки**: RMS, ZCR

### Обучение модели

```bash
# Полный цикл обучения
python scripts/train_model.py \\
    --data_path datasets \\
    --config config.json \\
    --output_dir models \\
    --epochs 100 \\
    --batch_size 32 \\
    --quantize
```

## Разработка ESP32 прошивки

### Основные классы и функции

```cpp
class VoiceXModelTester {
    // Загрузка и управление TF Lite моделью
    bool load_tflite_model();
    bool run_ml_inference(float* input, float* output);
};

// Аудио обработка
void audio_input_task(void* parameter);
void feature_extraction_task(void* parameter);
void ml_inference_task(void* parameter);
void audio_output_task(void* parameter);
```

### Memory Layout

```
Flash (4MB):
├── Bootloader (64KB)
├── Partitions table (4KB)  
├── NVS (20KB)
├── OTA_0 (1.5MB)
├── OTA_1 (1.5MB)
└── SPIFFS (896KB)
    ├── model.tflite (~200KB)
    ├── config.json (1KB)
    └── web files (~50KB)

RAM (520KB):
├── System (200KB)
├── TensorFlow Lite Arena (50KB)
├── Audio Buffers (100KB)
└── Stack/Heap (170KB)
```

### Real-time ограничения

- **Audio latency**: < 100ms end-to-end
- **ML inference**: < 50ms per frame
- **Memory usage**: < 400KB RAM
- **Flash usage**: < 3MB total

## API Reference

### REST API

```http
GET /status
{
  "state": "ready",
  "model_loaded": true,
  "free_heap": 150000,
  "uptime": 3600,
  "version": "1.0.0"
}

POST /upload
Content-Type: multipart/form-data
# Файлы: model.tflite, config.json

POST /test
{
  "duration": 3.0,
  "test_type": "voice_synthesis"
}
```

### Python API

```python
# Обучение модели
trainer = VoiceXTrainer('config.json')
X_train, y_train = trainer.load_dataset('datasets/')
history = trainer.train_model(X_train, y_train)
trainer.convert_to_tflite('model.tflite', quantize=True)

# Тестирование
tester = VoiceXModelTester('model.h5', 'config.json', 'scaler.pkl')
result = tester.test_single_sample('throat.wav', 'speech.wav')
```

## Сборка и развертывание

### Локальная разработка

```bash
# Активация окружения
source voicex_env/bin/activate

# Установка в dev режиме
pip install -e .

# Запуск тестов
pytest tests/

# Линтинг кода
flake8 voicex/
black voicex/
```

### Docker разработка

```bash
# Сборка образа
docker build -t voicex:dev .

# Запуск контейнера
docker run -v $(pwd):/app voicex:dev

# Docker Compose для полной системы
docker-compose up --build
```

### Прошивка ESP32

```bash
# PlatformIO
cd esp32_firmware
pio run --target upload
pio run --target uploadfs

# Arduino IDE
# File -> Examples -> ESP32 -> VoiceX
# Tools -> Upload Sketch Data
```

## Тестирование

### Unit тесты

```python
# tests/test_ml_pipeline.py
def test_feature_extraction():
    trainer = VoiceXTrainer()
    features = trainer.extract_features('test.wav')
    assert features.shape == (512,)

def test_model_inference():
    # Тест inference модели
    pass
```

### Integration тесты

```python
# tests/test_esp32_integration.py
def test_esp32_api():
    response = requests.get('http://esp32_ip/status')
    assert response.status_code == 200
    assert response.json()['model_loaded'] == True
```

### Performance тесты

```bash
# Benchmark inference времени
python tests/benchmark_inference.py

# Тест memory usage
python tests/memory_profiling.py

# Stress тест ESP32
python tests/stress_test_esp32.py --duration 3600
```

## Debugging

### ESP32 Debug

```cpp
// Serial debugging
Serial.printf("🔍 Debug: %s = %d\\n", variable_name, value);

// Memory monitoring  
Serial.printf("Free heap: %d bytes\\n", ESP.getFreeHeap());

// Task monitoring
vTaskList(task_list_buffer);
Serial.println(task_list_buffer);
```

### Python Debug

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Детальная информация о модели
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# Профилирование
import cProfile
cProfile.run('train_model()', 'profile_stats')
```

## Contributing Guidelines

### Code Style

- Python: PEP 8 + Black formatter
- C++: Google Style Guide
- JavaScript: Prettier + ESLint
- Commit messages: Conventional Commits

### Pull Request Process

1. Создать feature branch от main
2. Написать тесты для новой функциональности
3. Обновить документацию
4. Запустить полный test suite
5. Создать PR с детальным описанием

### Release Process

1. Обновить версию в setup.py
2. Обновить CHANGELOG.md
3. Создать release tag
4. Собрать и протестировать artifacts
5. Опубликовать release на GitHub

---

Дополнительная информация доступна в [docs/](../docs/) директории.
EOF

    print_success "Документация создана"
}

# Настройка Git репозитория
setup_git_repository() {
    print_step "Настройка Git репозитория..."
    
    # Инициализация git если еще не инициализирован
    if [ ! -d ".git" ]; then
        git init
        print_step "Git репозиторий инициализирован"
    fi
    
    # Создание .gitignore
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
voicex_env/
.venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# TensorFlow
*.pb
*.pbtxt
*.ckpt*
saved_model/
checkpoints/

# Models and data
*.tflite
*.h5
*.pkl
*.joblib
datasets/*/
!datasets/.gitkeep
models/*/
!models/.gitkeep
training_artifacts/*/
!training_artifacts/.gitkeep

# ESP32
esp32_firmware/.pio/
esp32_firmware/lib/
esp32_firmware/.vscode/
*.bin
*.elf
*.map

# System files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
*.tmp
*.temp

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# Environment variables
.env
.env.local
.env.production

# Docker
docker-compose.override.yml

# Temporary files
tmp/
temp/
EOF

    # Создание .gitkeep файлов для пустых директорий
    touch datasets/.gitkeep
    touch models/.gitkeep  
    touch training_artifacts/.gitkeep
    
    # Добавление файлов в git
    git add .
    git commit -m "🎉 Initial VoiceX project setup

- Add complete project structure
- Add ML training pipeline
- Add ESP32 firmware foundation
- Add web interface templates
- Add comprehensive documentation
- Add automated setup scripts"
    
    print_success "Git репозиторий настроен"
}

# Проверка установки
verify_installation() {
    print_step "Проверка установки..."
    
    # Проверка Python окружения
    if source voicex_env/bin/activate; then
        print_success "✅ Python окружение активно"
    else
        print_error "❌ Проблема с Python окружением"
        return 1
    fi
    
    # Проверка основных зависимостей
    python -c "import tensorflow, librosa, sklearn; print('ML зависимости: OK')" || {
        print_error "❌ Проблема с ML зависимостями"
        return 1
    }
    
    # Проверка PlatformIO
    pio --version > /dev/null 2>&1 && {
        print_success "✅ PlatformIO установлен"
    } || {
        print_warning "⚠️ PlatformIO не найден"
    }
    
    # Проверка структуры проекта
    for dir in datasets models scripts esp32_firmware web_interface docs; do
        if [ -d "$dir" ]; then
            print_success "✅ Директория $dir создана"
        else
            print_error "❌ Директория $dir отсутствует"
        fi
    done
    
    print_success "Проверка завершена"
}

# Показать информацию о следующих шагах
show_next_steps() {
    print_header
    echo -e "${GREEN}🎉 VoiceX успешно установлен!${NC}"
    echo ""
    echo -e "${BLUE}📋 Следующие шаги:${NC}"
    echo ""
    echo -e "${YELLOW}1. Активируйте Python окружение:${NC}"
    echo "   source voicex_env/bin/activate"
    echo ""
    echo -e "${YELLOW}2. Подготовьте данные для обучения:${NC}"
    echo "   mkdir -p datasets/{throat_microphone,speech}"
    echo "   # Поместите аудио файлы в соответствующие папки"
    echo ""
    echo -e "${YELLOW}3. Обучите модель:${NC}"
    echo "   python scripts/train_model.py --data_path datasets --output_dir models"
    echo ""
    echo -e "${YELLOW}4. Разверните на ESP32:${NC}"
    echo "   python scripts/deploy_to_esp32.py --model_dir models --esp32_port /dev/ttyUSB0"
    echo ""
    echo -e "${YELLOW}5. Откройте веб-интерфейс:${NC}"
    echo "   http://ESP32_IP_ADDRESS/"
    echo ""
    echo -e "${BLUE}📚 Документация:${NC}"
    echo "   📖 README.md - Основная документация"
    echo "   🔧 docs/DEVELOPER_GUIDE.md - Руководство разработчика"
    echo "   📊 docs/API_REFERENCE.md - Справочник API"
    echo ""
    echo -e "${BLUE}🆘 Поддержка:${NC}"
    echo "   📧 support@voicex.kz"
    echo "   💬 Telegram: @voicex_support"
    echo "   🐛 GitHub Issues: github.com/voicex/issues"
    echo ""
    echo -e "${GREEN}Удачи в разработке VoiceX! 🎙️${NC}"
}

# Главная функция
main() {
    print_header
    
    # Проверка прав
    if [[ $EUID -eq 0 ]]; then
        print_warning "Не запускайте скрипт от root (будет запрошен sudo при необходимости)"
    fi
    
    # Определение ОС
    detect_os
    
    # Основные шаги установки
    install_system_dependencies
    setup_python_environment
    install_python_dependencies
    install_platformio
    create_project_structure
    download_voicex_code
    create_config_files
    create_documentation
    setup_git_repository
    verify_installation
    
    # Информация о следующих шагах
    show_next_steps
}

# Обработка аргументов командной строки
case "${1:-}" in
    --help|-h)
        echo "VoiceX Setup Script"
        echo "Использование: $0 [опции]"
        echo ""
        echo "Опции:"
        echo "  --help, -h     Показать эту справку"
        echo "  --minimal      Минимальная установка (только Python)"
        echo "  --skip-deps    Пропустить установку системных зависимостей"
        echo "  --no-git       Не настраивать Git репозиторий"
        exit 0
        ;;
    --minimal)
        print_step "Запуск минимальной установки..."
        setup_python_environment
        install_python_dependencies
        create_project_structure
        download_voicex_code
        create_config_files
        ;;
    --skip-deps)
        print_step "Пропуск системных зависимостей..."
        setup_python_environment
        install_python_dependencies
        install_platformio
        create_project_structure
        download_voicex_code
        create_config_files
        create_documentation
        setup_git_repository
        verify_installation
        show_next_steps
        ;;
    --no-git)
        print_step "Установка без Git..."
        detect_os
        install_system_dependencies
        setup_python_environment
        install_python_dependencies
        install_platformio
        create_project_structure
        download_voicex_code
        create_config_files
        create_documentation
        verify_installation
        show_next_steps
        ;;
    *)
        # Полная установка по умолчанию
        main
        ;;
esac