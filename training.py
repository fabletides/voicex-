#!/usr/bin/env python3
"""
VoiceX Complete Training Pipeline
Полный пайплайн для обучения модели и генерации всех необходимых файлов
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import subprocess
import sys

# Импорт собственных модулей
try:
    from ml_pipeline import VoiceXTrainer
    from model_testing import VoiceXModelTester
    from esp32_utils import ESP32ModelUploader, create_deployment_package
    from model_converter import convert_tflite_to_header, create_demo_model
except ImportError as e:
    print(f"❌ Ошибка импорта модулей: {e}")
    print("💡 Убедитесь, что все файлы находятся в одной директории")
    sys.exit(1)

class VoiceXPipeline:
    """Главный класс для управления полным пайплайном VoiceX"""
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.output_dir = Path("pipeline_output")
        
        # Создание директорий
        self.create_directory_structure()
        
        print("🚀 VoiceX Complete Pipeline инициализирован")
    
    def load_config(self):
        """Загрузка конфигурации пайплайна"""
        default_config = {
            "pipeline": {
                "create_demo_data": True,
                "train_model": True,
                "test_model": True,
                "convert_to_tflite": True,
                "create_header": True,
                "create_spiffs_data": True,
                "create_deployment_package": True
            },
            "model": {
                "input_dim": 512,
                "output_dim": 256,
                "epochs": 50,
                "batch_size": 32,
                "quantize": True
            },
            "esp32": {
                "create_platformio_project": True,
                "copy_firmware": True,
                "generate_build_scripts": True
            },
            "demo": {
                "num_samples": 100,
                "sample_rate": 22050,
                "duration": 3.0
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
            # Объединяем конфигурации
            for key in default_config:
                if key in user_config:
                    default_config[key].update(user_config[key])
        else:
            # Создаем дефолтный конфиг
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"📋 Создан дефолтный конфиг: {self.config_path}")
        
        return default_config
    
    def create_directory_structure(self):
        """Создание структуры директорий"""
        dirs = [
            "pipeline_output",
            "pipeline_output/models",
            "pipeline_output/datasets", 
            "pipeline_output/esp32_firmware",
            "pipeline_output/spiffs_data",
            "pipeline_output/deployment",
            "pipeline_output/tests",
            "pipeline_output/logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def step1_create_demo_data(self):
        """Шаг 1: Создание демо данных для тестирования"""
        if not self.config['pipeline']['create_demo_data']:
            print("⏭️ Пропуск создания демо данных")
            return True
        
        print("📊 Шаг 1: Создание демо данных...")
        
        try:
            import numpy as np
            import soundfile as sf
            import librosa
            
            datasets_dir = self.output_dir / "datasets"
            throat_dir = datasets_dir / "throat_microphone"
            speech_dir = datasets_dir / "speech"
            
            throat_dir.mkdir(exist_ok=True)
            speech_dir.mkdir(exist_ok=True)
            
            # Параметры
            sr = self.config['demo']['sample_rate']
            duration = self.config['demo']['duration']
            num_samples = self.config['demo']['num_samples']
            
            print(f"   Создание {num_samples} демо образцов...")
            
            for i in range(num_samples):
                # Создание синтетических данных
                t = np.linspace(0, duration, int(sr * duration))
                
                # Сигнал горлового микрофона (зашумленный, приглушенный)
                throat_signal = 0.3 * np.sin(2 * np.pi * 150 * t) + \
                               0.2 * np.sin(2 * np.pi * 300 * t) + \
                               0.1 * np.random.normal(0, 0.1, len(t))
                
                # Нормальная речь (более чистая, с гармониками)
                speech_signal = 0.5 * np.sin(2 * np.pi * 150 * t) + \
                               0.3 * np.sin(2 * np.pi * 300 * t) + \
                               0.2 * np.sin(2 * np.pi * 450 * t) + \
                               0.1 * np.sin(2 * np.pi * 600 * t) + \
                               0.05 * np.random.normal(0, 0.05, len(t))
                
                # Применение окна для сглаживания
                window = np.hanning(len(t))
                throat_signal *= window
                speech_signal *= window
                
                # Сохранение файлов
                throat_file = throat_dir / f"sample_{i:03d}.wav"
                speech_file = speech_dir / f"sample_{i:03d}.wav"
                
                sf.write(throat_file, throat_signal, sr)
                sf.write(speech_file, speech_signal, sr)
            
            print(f"   ✅ Создано {num_samples} пар демо файлов")
            print(f"   📁 Горловой микрофон: {throat_dir}")
            print(f"   📁 Обычная речь: {speech_dir}")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Не удалось создать демо данные: {e}")
            print("   💡 Установите: pip install numpy soundfile librosa")
            return False
        except Exception as e:
            print(f"   ❌ Ошибка создания демо данных: {e}")
            return False
    
    def step2_train_model(self):
        """Шаг 2: Обучение ML модели"""
        if not self.config['pipeline']['train_model']:
            print("⏭️ Пропуск обучения модели")
            return True
        
        print("🧠 Шаг 2: Обучение модели...")
        
        try:
            # Инициализация тренера
            trainer = VoiceXTrainer()
            
            # Загрузка данных
            data_path = self.output_dir / "datasets"
            if not data_path.exists():
                print("   ❌ Данные для обучения не найдены")
                return False
            
            print("   📂 Загрузка данных...")
            X_data, y_data = trainer.load_dataset(str(data_path))
            
            if X_data is None or len(X_data) == 0:
                print("   ❌ Не удалось загрузить данные")
                return False
            
            print(f"   📊 Загружено {len(X_data)} образцов")
            
            # Разделение данных
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.2, random_state=42
            )
            
            # Обучение
            print("   🎯 Начало обучения...")
            history = trainer.train_model(X_train, y_train)
            
            # Сохранение артефактов
            models_dir = self.output_dir / "models"
            trainer.save_training_artifacts(str(models_dir))
            
            # Конвертация в TFLite
            if self.config['pipeline']['convert_to_tflite']:
                print("   🔄 Конвертация в TensorFlow Lite...")
                tflite_path = models_dir / "voicex_model.tflite"
                
                quantize = self.config['model']['quantize']
                trainer.convert_to_tflite(
                    str(tflite_path), 
                    quantize=quantize,
                    X_data=X_train if quantize else None
                )
            
            print("   ✅ Обучение модели завершено")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка обучения модели: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step3_test_model(self):
        """Шаг 3: Тестирование модели"""
        if not self.config['pipeline']['test_model']:
            print("⏭️ Пропуск тестирования модели")
            return True
        
        print("🧪 Шаг 3: Тестирование модели...")
        
        try:
            models_dir = self.output_dir / "models"
            model_path = models_dir / "voicex_model.h5"
            config_path = models_dir / "config.json"
            scaler_path = models_dir / "scaler.pkl"
            
            if not all(p.exists() for p in [model_path, config_path, scaler_path]):
                print("   ❌ Не все файлы модели найдены")
                return False
            
            # Инициализация тестера
            tester = VoiceXModelTester(
                str(model_path),
                str(config_path), 
                str(scaler_path)
            )
            
            # Тестирование на демо данных
            test_dir = self.output_dir / "tests"
            datasets_dir = self.output_dir / "datasets"
            
            results = tester.test_dataset(
                str(datasets_dir),
                str(test_dir),
                num_samples=10
            )
            
            if results:
                print("   ✅ Тестирование модели завершено")
                print(f"   📊 Результаты сохранены в {test_dir}")
                return True
            else:
                print("   ❌ Тестирование не удалось")
                return False
                
        except Exception as e:
            print(f"   ❌ Ошибка тестирования: {e}")
            return False
    
    def step4_create_header_file(self):
        """Шаг 4: Создание C++ заголовочного файла"""
        if not self.config['pipeline']['create_header']:
            print("⏭️ Пропуск создания заголовочного файла")
            return True
        
        print("🔧 Шаг 4: Создание C++ заголовочного файла...")
        
        try:
            models_dir = self.output_dir / "models"
            tflite_path = models_dir / "voicex_model.tflite"
            
            if not tflite_path.exists():
                print("   ⚠️ TFLite модель не найдена, создаем демо модель...")
                demo_model = create_demo_model()
                if demo_model:
                    shutil.move(demo_model, tflite_path)
                else:
                    print("   ❌ Не удалось создать демо модель")
                    return False
            
            # Конвертация в заголовочный файл
            header_path = models_dir / "voicex_model_data.h"
            convert_tflite_to_header(
                str(tflite_path),
                str(header_path),
                "voicex_model"
            )
            
            print(f"   ✅ Заголовочный файл создан: {header_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка создания заголовочного файла: {e}")
            return False
    
    def step5_create_spiffs_data(self):
        """Шаг 5: Подготовка данных для SPIFFS"""
        if not self.config['pipeline']['create_spiffs_data']:
            print("⏭️ Пропуск подготовки SPIFFS данных")
            return True
        
        print("📦 Шаг 5: Подготовка данных для SPIFFS...")
        
        try:
            models_dir = self.output_dir / "models"
            spiffs_dir = self.output_dir / "spiffs_data"
            
            # Создание загрузчика
            uploader = ESP32ModelUploader()
            
            # Подготовка данных
            uploader.prepare_spiffs_data(str(models_dir), str(spiffs_dir))
            
            print(f"   ✅ SPIFFS данные подготовлены в {spiffs_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка подготовки SPIFFS данных: {e}")
            return False
    
    def step6_create_esp32_project(self):
        """Шаг 6: Создание проекта ESP32"""
        if not self.config['esp32']['create_platformio_project']:
            print("⏭️ Пропуск создания ESP32 проекта")
            return True
        
        print("🔩 Шаг 6: Создание ESP32 проекта...")
        
        try:
            esp32_dir = self.output_dir / "esp32_firmware"
            
            # Создание PlatformIO конфигурации
            platformio_ini = """[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino

lib_deps = 
    bblanchon/ArduinoJson@^6.21.3
    https://github.com/tensorflow/tflite-micro-arduino-examples

build_flags = 
    -DCORE_DEBUG_LEVEL=3
    -DVOICEX_DEBUG=1
    -std=gnu++17

board_build.partitions = huge_app.csv
board_build.filesystem = spiffs

monitor_speed = 115200
upload_speed = 921600
"""
            
            with open(esp32_dir / "platformio.ini", 'w') as f:
                f.write(platformio_ini)
            
            # Создание структуры проекта
            (esp32_dir / "src").mkdir(exist_ok=True)
            (esp32_dir / "data").mkdir(exist_ok=True)
            (esp32_dir / "lib").mkdir(exist_ok=True)
            (esp32_dir / "include").mkdir(exist_ok=True)
            
            # Копирование SPIFFS данных
            spiffs_src = self.output_dir / "spiffs_data"
            spiffs_dst = esp32_dir / "data"
            
            if spiffs_src.exists():
                for file in spiffs_src.glob("*"):
                    if file.is_file():
                        shutil.copy2(file, spiffs_dst)
            
            # Копирование заголовочного файла модели
            models_dir = self.output_dir / "models"
            header_file = models_dir / "voicex_model_data.h"
            if header_file.exists():
                shutil.copy2(header_file, esp32_dir / "include")
            
            print(f"   ✅ ESP32 проект создан в {esp32_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка создания ESP32 проекта: {e}")
            return False
    
    def step7_create_deployment_package(self):
        """Шаг 7: Создание пакета развертывания"""
        if not self.config['pipeline']['create_deployment_package']:
            print("⏭️ Пропуск создания пакета развертывания")
            return True
        
        print("📦 Шаг 7: Создание пакета развертывания...")
        
        try:
            models_dir = self.output_dir / "models"
            deployment_dir = self.output_dir / "deployment"
            
            # Создание ZIP пакета
            package_path = create_deployment_package(
                str(models_dir),
                str(deployment_dir / "voicex_deployment.zip")
            )
            
            # Создание README для развертывания
            readme_content = """# VoiceX Deployment Package

## Быстрое развертывание

### 1. Прошивка ESP32
```bash
cd esp32_firmware
pio run --target upload
pio run --target uploadfs
```

### 2. Проверка работы
- Подключитесь к Serial Monitor (115200 baud)
- Должны увидеть: "🎉 VoiceX система готова к работе!"

### 3. Веб-интерфейс
- Найдите IP адрес ESP32 в Serial Monitor
- Откройте http://ESP32_IP в браузере

## Файлы в пакете
- voicex_model.tflite - обученная модель
- config.json - конфигурация
- scaler.pkl - нормализация данных
- voicex_model_data.h - встроенная модель для C++

## Поддержка
- Email: support@voicex.kz
- Telegram: @voicex_support
"""
            
            with open(deployment_dir / "README.md", 'w') as f:
                f.write(readme_content)
            
            print(f"   ✅ Пакет развертывания создан в {deployment_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка создания пакета развертывания: {e}")
            return False
    
    def step8_generate_build_scripts(self):
        """Шаг 8: Генерация скриптов сборки"""
        print("📜 Шаг 8: Генерация скриптов сборки...")
        
        try:
            scripts_dir = self.output_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            
            # Скрипт сборки и загрузки
            build_script = """#!/bin/bash
# VoiceX Build and Deploy Script

set -e

echo "🔨 Сборка VoiceX ESP32 прошивки..."

cd esp32_firmware

# Проверка PlatformIO
if ! command -v pio &> /dev/null; then
    echo "❌ PlatformIO не найден"
    echo "💡 Установите: pip install platformio"
    exit 1
fi

# Сборка прошивки
echo "🔧 Компиляция..."
pio run

# Загрузка прошивки
echo "📡 Загрузка прошивки..."
pio run --target upload

# Загрузка файловой системы
echo "📁 Загрузка SPIFFS..."
pio run --target uploadfs

echo "✅ Развертывание завершено!"
echo "🔍 Откройте Serial Monitor для проверки:"
echo "   pio device monitor"
"""
            
            with open(scripts_dir / "build_and_deploy.sh", 'w') as f:
                f.write(build_script)
            
            # Скрипт тестирования
            test_script = """#!/bin/bash
# VoiceX Test Script

echo "🧪 Тестирование VoiceX системы..."

# Проверка подключения ESP32
if [ "$1" ]; then
    ESP32_IP=$1
else
    echo "💡 Использование: $0 <ESP32_IP>"
    echo "💡 Например: $0 192.168.1.100"
    exit 1
fi

# Проверка статуса
echo "🔍 Проверка статуса ESP32..."
response=$(curl -s http://$ESP32_IP/status || echo "error")

if [ "$response" == "error" ]; then
    echo "❌ ESP32 не отвечает по адресу $ESP32_IP"
    exit 1
fi

echo "✅ ESP32 отвечает"
echo "$response" | python -m json.tool

# Тест вибромоторов
echo "📳 Тест вибромоторов..."
curl -s -X POST http://$ESP32_IP/test

echo "✅ Тестирование завершено!"
"""
            
            with open(scripts_dir / "test_system.sh", 'w') as f:
                f.write(test_script)
            
            # Скрипт мониторинга
            monitor_script = """#!/bin/bash
# VoiceX Monitor Script

ESP32_IP=${1:-"192.168.1.100"}

echo "📊 Мониторинг VoiceX системы: $ESP32_IP"
echo "Нажмите Ctrl+C для выхода"

while true; do
    clear
    echo "🎙️ VoiceX System Monitor - $(date)"
    echo "================================"
    
    status=$(curl -s http://$ESP32_IP/status 2>/dev/null || echo '{"error": "no connection"}')
    
    if echo "$status" | grep -q "error"; then
        echo "❌ Нет подключения к ESP32"
    else
        echo "$status" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Состояние: {data.get(\"state\", \"неизвестно\")}')
    print(f'Модель: {\"✅ Загружена\" if data.get(\"model_loaded\") else \"❌ Не загружена\"}')
    print(f'Память: {data.get(\"free_heap\", 0):,} байт')
    print(f'Время работы: {data.get(\"uptime\", 0)} сек')
except:
    print('Ошибка парсинга ответа')
"
    fi
    
    echo ""
    echo "🔄 Обновление через 5 секунд..."
    sleep 5
done
"""
            
            with open(scripts_dir / "monitor.sh", 'w') as f:
                f.write(monitor_script)
            
            # Делаем скрипты исполняемыми
            for script in scripts_dir.glob("*.sh"):
                os.chmod(script, 0o755)
            
            print(f"   ✅ Скрипты созданы в {scripts_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка создания скриптов: {e}")
            return False
    
    def generate_summary_report(self):
        """Генерация итогового отчета"""
        print("📋 Генерация итогового отчета...")
        
        report_content = f"""
# VoiceX Pipeline Execution Report
Отчет о выполнении полного пайплайна VoiceX

## Конфигурация
- Входная размерность: {self.config['model']['input_dim']}
- Выходная размерность: {self.config['model']['output_dim']}
- Эпохи обучения: {self.config['model']['epochs']}
- Размер батча: {self.config['model']['batch_size']}
- Квантизация: {self.config['model']['quantize']}

## Сгенерированные файлы

### Модели
- `models/voicex_model.h5` - Обученная Keras модель
- `models/voicex_model.tflite` - TensorFlow Lite модель
- `models/voicex_model_data.h` - C++ заголовочный файл
- `models/scaler.pkl` - Параметры нормализации
- `models/config.json` - Конфигурация модели

### ESP32 Проект
- `esp32_firmware/` - Полный проект PlatformIO
- `esp32_firmware/src/main.cpp` - Основной код прошивки
- `esp32_firmware/data/` - Файлы для SPIFFS
- `esp32_firmware/platformio.ini` - Конфигурация сборки

### Данные и тесты
- `datasets/` - Демо данные для обучения
- `tests/` - Результаты тестирования модели
- `spiffs_data/` - Подготовленные данные для SPIFFS

### Развертывание
- `deployment/voicex_deployment.zip` - Пакет развертывания
- `scripts/build_and_deploy.sh` - Скрипт сборки и загрузки
- `scripts/test_system.sh` - Скрипт тестирования
- `scripts/monitor.sh` - Скрипт мониторинга

## Следующие шаги

### 1. Сборка и загрузка прошивки
```bash
cd {self.output_dir}/esp32_firmware
pio run --target upload
pio run --target uploadfs
```

### 2. Проверка работы
```bash
pio device monitor
# Должно появиться: "🎉 VoiceX система готова к работе!"
```

### 3. Веб-интерфейс
- Найдите IP адрес ESP32 в Serial Monitor
- Откройте http://ESP32_IP в браузере

### 4. Тестирование системы
```bash
{self.output_dir}/scripts/test_system.sh <ESP32_IP>
```

### 5. Мониторинг
```bash
{self.output_dir}/scripts/monitor.sh <ESP32_IP>
```

## Устранение неполадок

### Модель не загружается
- Проверьте размер модели (должен быть < 500KB)
- Убедитесь, что SPIFFS данные загружены
- Проверьте логи в Serial Monitor

### ESP32 не отвечает
- Проверьте подключение к WiFi
- Убедитесь, что прошивка загружена корректно
- Перезагрузите ESP32

### Ошибки компиляции
- Обновите PlatformIO: `pio upgrade`
- Очистите кэш: `pio run --target clean`
- Проверьте доступность библиотек

## Поддержка
- Email: support@voicex.kz
- Telegram: @voicex_support
- GitHub: github.com/voicex/issues

---
Отчет сгенерирован: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_path = self.output_dir / "PIPELINE_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   ✅ Отчет сохранен: {report_path}")
        return report_path
    
    def run_full_pipeline(self):
        """Запуск полного пайплайна"""
        print("🚀 Запуск полного пайплайна VoiceX")
        print("=" * 50)
        
        steps = [
            ("Создание демо данных", self.step1_create_demo_data),
            ("Обучение модели", self.step2_train_model),
            ("Тестирование модели", self.step3_test_model),
            ("Создание C++ заголовка", self.step4_create_header_file),
            ("Подготовка SPIFFS", self.step5_create_spiffs_data),
            ("Создание ESP32 проекта", self.step6_create_esp32_project),
            ("Создание пакета развертывания", self.step7_create_deployment_package),
            ("Генерация скриптов", self.step8_generate_build_scripts)
        ]
        
        successful_steps = 0
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                print(f"\n📋 {step_name}...")
                if step_func():
                    successful_steps += 1
                    print(f"✅ {step_name} - УСПЕШНО")
                else:
                    failed_steps.append(step_name)
                    print(f"❌ {step_name} - ОШИБКА")
            except Exception as e:
                failed_steps.append(step_name)
                print(f"❌ {step_name} - ИСКЛЮЧЕНИЕ: {e}")
        
        # Генерация отчета
        print(f"\n📋 Генерация итогового отчета...")
        report_path = self.generate_summary_report()
        
        # Итоговый статус
        print("\n" + "=" * 50)
        print("🏁 Пайплайн завершен!")
        print(f"✅ Успешных шагов: {successful_steps}/{len(steps)}")
        
        if failed_steps:
            print(f"❌ Неудачных шагов: {len(failed_steps)}")
            for step in failed_steps:
                print(f"   - {step}")
        else:
            print("🎉 Все шаги выполнены успешно!")
        
        print(f"\n📁 Результаты в: {self.output_dir}")
        print(f"📋 Полный отчет: {report_path}")
        
        return len(failed_steps) == 0

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='VoiceX Complete Pipeline')
    parser.add_argument('--config', type=str, default='pipeline_config.json',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--output_dir', type=str, default='pipeline_output',
                       help='Директория для результатов')
    parser.add_argument('--skip_demo_data', action='store_true',
                       help='Пропустить создание демо данных')
    parser.add_argument('--skip_training', action='store_true',
                       help='Пропустить обучение модели')
    parser.add_argument('--demo_only', action='store_true',
                       help='Только демо модель без обучения')
    
    args = parser.parse_args()
    
    print("🎙️ VoiceX Complete Training Pipeline")
    print("=" * 50)
    
    try:
        # Инициализация пайплайна
        pipeline = VoiceXPipeline(args.config)
        
        # Переопределение директории вывода
        if args.output_dir != 'pipeline_output':
            pipeline.output_dir = Path(args.output_dir)
            pipeline.create_directory_structure()
        
        # Настройка конфигурации на основе аргументов
        if args.skip_demo_data:
            pipeline.config['pipeline']['create_demo_data'] = False
        
        if args.skip_training:
            pipeline.config['pipeline']['train_model'] = False
            pipeline.config['pipeline']['test_model'] = False
        
        if args.demo_only:
            # Только демо без полного обучения
            pipeline.config['pipeline']['train_model'] = False
            pipeline.config['pipeline']['test_model'] = False
            pipeline.config['pipeline']['create_demo_data'] = True
        
        # Запуск пайплайна
        success = pipeline.run_full_pipeline()
        
        if success:
            print("\n🎉 Пайплайн выполнен успешно!")
            print("\n📝 Следующие шаги:")
            print("1. cd pipeline_output/esp32_firmware")
            print("2. pio run --target upload")
            print("3. pio run --target uploadfs")
            print("4. pio device monitor")
            return 0
        else:
            print("\n❌ Пайплайн завершен с ошибками")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Пайплайн прерван пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Критическая ошибка пайплайна: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())