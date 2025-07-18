#!/usr/bin/env python3
"""
VoiceX ESP32 Utilities
Утилиты для подготовки и загрузки модели в ESP32
"""

import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path
import requests
import zipfile
import tempfile

class ESP32ModelUploader:
    """Класс для загрузки модели в ESP32"""
    
    def __init__(self, esp32_ip=None, esp32_port="/dev/ttyUSB0"):
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        
    def prepare_spiffs_data(self, model_dir, spiffs_dir="data"):
        """Подготовка данных для SPIFFS"""
        print(f"📦 Подготовка данных SPIFFS в {spiffs_dir}...")
        
        # Создание директории SPIFFS
        os.makedirs(spiffs_dir, exist_ok=True)
        
        model_dir = Path(model_dir)
        spiffs_dir = Path(spiffs_dir)
        
        # Копирование необходимых файлов
        files_to_copy = [
            ("voicex_model.tflite", "model.tflite"),
            ("esp32_config.json", "config.json"),
            ("scaler.pkl", "scaler.pkl")
        ]
        
        for src_name, dst_name in files_to_copy:
            src_path = model_dir / src_name
            dst_path = spiffs_dir / dst_name
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"   ✅ {src_name} → {dst_name}")
            else:
                print(f"   ⚠️ Файл не найден: {src_name}")
        
        # Создание файла манифеста
        manifest = {
            "version": "1.0",
            "files": [dst_name for _, dst_name in files_to_copy if (spiffs_dir / dst_name).exists()],
            "total_size": sum((spiffs_dir / dst_name).stat().st_size 
                             for _, dst_name in files_to_copy if (spiffs_dir / dst_name).exists()),
            "description": "VoiceX ML Model and Configuration"
        }
        
        with open(spiffs_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📋 Манифест создан: {manifest['total_size']} байт")
        return spiffs_dir
    
    def upload_via_platformio(self, spiffs_dir="data"):
        """Загрузка через PlatformIO"""
        print("📡 Загрузка файлов через PlatformIO...")
        
        try:
            # Проверка наличия PlatformIO
            result = subprocess.run(['pio', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ PlatformIO не найден. Установите: pip install platformio")
                return False
            
            print(f"✅ PlatformIO обнаружен: {result.stdout.strip()}")
            
            # Загрузка файловой системы
            cmd = ['pio', 'run', '--target', 'uploadfs']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Файлы успешно загружены в SPIFFS")
                return True
            else:
                print(f"❌ Ошибка загрузки: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ PlatformIO не найден")
            return False
    
    def upload_via_web_interface(self, spiffs_dir="data"):
        """Загрузка через веб-интерфейс ESP32"""
        if not self.esp32_ip:
            print("❌ IP адрес ESP32 не указан")
            return False
        
        print(f"🌐 Загрузка через веб-интерфейс: http://{self.esp32_ip}")
        
        spiffs_dir = Path(spiffs_dir)
        
        # Загрузка каждого файла
        for file_path in spiffs_dir.glob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        files = {'file': (file_path.name, f)}
                        response = requests.post(
                            f"http://{self.esp32_ip}/upload",
                            files=files,
                            timeout=30
                        )
                    
                    if response.status_code == 200:
                        print(f"   ✅ {file_path.name}")
                    else:
                        print(f"   ❌ {file_path.name}: {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ {file_path.name}: {e}")
        
        return True
    
    def create_arduino_header(self, model_path, output_path="model_data.h"):
        """Создание C++ заголовочного файла с данными модели"""
        print(f"🔧 Создание заголовочного файла: {output_path}")
        
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Создание C++ массива
        header_content = f"""
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Автоматически сгенерированный файл с данными модели VoiceX
// Размер модели: {len(model_data)} байт

const unsigned char voicex_model_data[] = {{
"""
        
        # Добавление байтов модели
        for i, byte in enumerate(model_data):
            if i % 16 == 0:
                header_content += "\n  "
            header_content += f"0x{byte:02x}, "
        
        header_content += f"""
}};

const unsigned int voicex_model_data_len = {len(model_data)};

#endif // MODEL_DATA_H
"""
        
        with open(output_path, 'w') as f:
            f.write(header_content)
        
        print(f"✅ Заголовочный файл создан: {len(model_data)} байт")
        return output_path

class PlatformIOProject:
    """Класс для управления проектом PlatformIO"""
    
    def __init__(self, project_dir="voicex_esp32"):
        self.project_dir = Path(project_dir)
    
    def create_project(self):
        """Создание проекта PlatformIO"""
        print(f"🏗️ Создание проекта PlatformIO: {self.project_dir}")
        
        # Создание директории проекта
        self.project_dir.mkdir(exist_ok=True)
        
        # Создание platformio.ini
        platformio_ini = """
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino

; Настройки для VoiceX
build_flags = 
    -DCORE_DEBUG_LEVEL=3
    -DVOICEX_DEBUG=1

; Библиотеки
lib_deps = 
    bblanchon/ArduinoJson@^6.21.3
    https://github.com/tensorflow/tflite-micro-arduino-examples.git

; Настройки SPIFFS
board_build.partitions = huge_app.csv
board_build.filesystem = spiffs

; Настройки загрузки
monitor_speed = 115200
upload_speed = 921600

; Настройки компиляции
build_unflags = -std=gnu++11
build_flags = -std=gnu++17
"""
        
        with open(self.project_dir / "platformio.ini", 'w') as f:
            f.write(platformio_ini)
        
        # Создание структуры папок
        (self.project_dir / "src").mkdir(exist_ok=True)
        (self.project_dir / "data").mkdir(exist_ok=True)
        (self.project_dir / "lib").mkdir(exist_ok=True)
        (self.project_dir / "include").mkdir(exist_ok=True)
        
        print("✅ Проект PlatformIO создан")
        return self.project_dir
    
    def copy_source_files(self, voicex_cpp_path):
        """Копирование исходных файлов"""
        src_dir = self.project_dir / "src"
        
        if Path(voicex_cpp_path).exists():
            shutil.copy2(voicex_cpp_path, src_dir / "main.cpp")
            print("✅ Основной код скопирован")
        else:
            print("⚠️ Основной файл кода не найден")

def create_deployment_package(model_dir, output_path="voicex_deployment.zip"):
    """Создание пакета для развертывания"""
    print(f"📦 Создание пакета развертывания: {output_path}")
    
    model_dir = Path(model_dir)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Добавление файлов модели
        for file_path in model_dir.glob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.name)
                print(f"   + {file_path.name}")
        
        # Добавление README
        readme_content = """
# VoiceX Deployment Package

## Содержимое пакета:
- voicex_model.tflite - обученная модель TensorFlow Lite
- esp32_config.json - конфигурация для ESP32
- scaler.pkl - параметры нормализации данных
- voicex_model.h5 - исходная модель Keras (для справки)
- training_history.png - график обучения

## Установка на ESP32:

### Метод 1: PlatformIO
1. Скопируйте файлы в папку data/ вашего проекта
2. Выполните: `pio run --target uploadfs`

### Метод 2: Arduino IDE
1. Установите ESP32 Sketch Data Upload
2. Скопируйте файлы в папку data/
3. Используйте Tools -> ESP32 Sketch Data Upload

### Метод 3: Веб-интерфейс
1. Подключитесь к веб-интерфейсу ESP32
2. Загрузите файлы через форму

## Проверка:
После загрузки проверьте Serial Monitor:
- ✅ Модель загружена: /model.tflite
- ✅ Конфигурация загружена
- 📊 Тип модели: tflite
- 🎯 VoiceX система готова к работе!
"""
        
        # Добавляем README в архив
        zipf.writestr("README.md", readme_content)
        
    print(f"✅ Пакет создан: {output_path}")
    return output_path

def verify_esp32_model(esp32_ip=None, esp32_port="/dev/ttyUSB0"):
    """Проверка загруженной модели на ESP32"""
    print("🔍 Проверка модели на ESP32...")
    
    if esp32_ip:
        # Проверка через HTTP API
        try:
            response = requests.get(f"http://{esp32_ip}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print("✅ ESP32 отвечает:")
                print(f"   Состояние: {status.get('state', 'неизвестно')}")
                print(f"   Модель: {status.get('model_loaded', 'не загружена')}")
                print(f"   Память: {status.get('free_heap', 'неизвестно')} байт")
                return True
            else:
                print(f"❌ ESP32 не отвечает: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Ошибка подключения к ESP32: {e}")
            return False
    else:
        print("⚠️ IP адрес ESP32 не указан, проверьте Serial Monitor")
        return True

def flash_esp32_firmware(firmware_path, esp32_port="/dev/ttyUSB0"):
    """Прошивка ESP32"""
    print(f"⚡ Прошивка ESP32 через {esp32_port}...")
    
    try:
        # Используем esptool для прошивки
        cmd = [
            'esptool.py',
            '--port', esp32_port,
            '--baud', '921600',
            'write_flash',
            '0x1000', firmware_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Прошивка успешна")
            return True
        else:
            print(f"❌ Ошибка прошивки: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ esptool.py не найден. Установите: pip install esptool")
        return False

def create_web_interface_files(output_dir="web_interface"):
    """Создание файлов веб-интерфейса для ESP32"""
    print(f"🌐 Создание веб-интерфейса в {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # HTML страница для загрузки файлов
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>VoiceX ESP32 Interface</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .status { padding: 15px; margin: 20px 0; border-radius: 5px; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #007bff; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .file-list { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ VoiceX ESP32 Control Panel</h1>
        
        <div id="status" class="status">
            Состояние: <span id="statusText">Проверка...</span>
        </div>
        
        <h2>📁 Загрузка файлов модели</h2>
        <div class="upload-area" id="uploadArea">
            <p>Перетащите файлы сюда или нажмите для выбора</p>
            <input type="file" id="fileInput" multiple style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Выбрать файлы</button>
        </div>
        
        <div id="fileList" class="file-list" style="display: none;">
            <h3>Выбранные файлы:</h3>
            <ul id="files"></ul>
            <button onclick="uploadFiles()">📤 Загрузить в ESP32</button>
        </div>
        
        <h2>🎛️ Управление системой</h2>
        <button onclick="restartSystem()">🔄 Перезапуск</button>
        <button onclick="testVoice()">🎤 Тест голоса</button>
        <button onclick="downloadLogs()">📋 Скачать логи</button>
        
        <h2>📊 Системная информация</h2>
        <div id="systemInfo">
            <p>Загрузка...</p>
        </div>
    </div>

    <script>
        // Проверка статуса ESP32
        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('statusText').textContent = data.state || 'Неизвестно';
                    document.getElementById('status').className = 
                        'status ' + (data.model_loaded ? 'success' : 'error');
                    updateSystemInfo(data);
                })
                .catch(error => {
                    document.getElementById('statusText').textContent = 'Ошибка подключения';
                    document.getElementById('status').className = 'status error';
                });
        }
        
        function updateSystemInfo(data) {
            document.getElementById('systemInfo').innerHTML = `
                <p><strong>Модель загружена:</strong> ${data.model_loaded ? 'Да' : 'Нет'}</p>
                <p><strong>Свободная память:</strong> ${data.free_heap} байт</p>
                <p><strong>Время работы:</strong> ${data.uptime} сек</p>
                <p><strong>Версия:</strong> ${data.version || 'Неизвестно'}</p>
            `;
        }
        
        // Обработка файлов
        document.getElementById('fileInput').addEventListener('change', handleFiles);
        
        function handleFiles(event) {
            const files = event.target.files;
            if (files.length > 0) {
                const fileList = document.getElementById('files');
                fileList.innerHTML = '';
                
                for (let file of files) {
                    const li = document.createElement('li');
                    li.textContent = `${file.name} (${file.size} байт)`;
                    fileList.appendChild(li);
                }
                
                document.getElementById('fileList').style.display = 'block';
            }
        }
        
        function uploadFiles() {
            const files = document.getElementById('fileInput').files;
            const formData = new FormData();
            
            for (let file of files) {
                formData.append('files', file);
            }
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                alert(data.success ? 'Файлы загружены успешно!' : 'Ошибка загрузки: ' + data.error);
                checkStatus();
            })
            .catch(error => alert('Ошибка: ' + error));
        }
        
        function restartSystem() {
            if (confirm('Перезапустить систему?')) {
                fetch('/restart', { method: 'POST' })
                    .then(() => {
                        alert('Система перезапускается...');
                        setTimeout(checkStatus, 5000);
                    });
            }
        }
        
        function testVoice() {
            fetch('/test', { method: 'POST' })
                .then(response => response.json())
                .then(data => alert(data.message || 'Тест выполнен'));
        }
        
        function downloadLogs() {
            window.open('/logs', '_blank');
        }
        
        // Проверка статуса при загрузке и каждые 5 секунд
        checkStatus();
        setInterval(checkStatus, 5000);
    </script>
</body>
</html>
"""
    
    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Веб-интерфейс создан")
    return output_dir

def main():
    """Основная функция утилит"""
    parser = argparse.ArgumentParser(description='VoiceX ESP32 Utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['prepare', 'upload', 'verify', 'package', 'project', 'web'],
                       help='Действие для выполнения')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Директория с моделью')
    parser.add_argument('--spiffs_dir', type=str, default='./data',
                       help='Директория для SPIFFS данных')
    parser.add_argument('--esp32_ip', type=str,
                       help='IP адрес ESP32 для веб-загрузки')
    parser.add_argument('--esp32_port', type=str, default='/dev/ttyUSB0',
                       help='Порт ESP32 для последовательной загрузки')
    parser.add_argument('--output', type=str,
                       help='Путь для вывода (пакет, проект и т.д.)')
    parser.add_argument('--cpp_file', type=str,
                       help='Путь к файлу main.cpp для проекта PlatformIO')
    
    args = parser.parse_args()
    
    print("🔧 VoiceX ESP32 Utilities")
    print("=" * 40)
    
    uploader = ESP32ModelUploader(args.esp32_ip, args.esp32_port)
    
    if args.action == 'prepare':
        # Подготовка данных SPIFFS
        spiffs_dir = uploader.prepare_spiffs_data(args.model_dir, args.spiffs_dir)
        print(f"✅ Данные подготовлены в {spiffs_dir}")
        
    elif args.action == 'upload':
        # Загрузка в ESP32
        spiffs_dir = uploader.prepare_spiffs_data(args.model_dir, args.spiffs_dir)
        
        if args.esp32_ip:
            success = uploader.upload_via_web_interface(spiffs_dir)
        else:
            success = uploader.upload_via_platformio(spiffs_dir)
        
        if success:
            print("✅ Загрузка завершена")
        else:
            print("❌ Ошибка загрузки")
            
    elif args.action == 'verify':
        # Проверка модели на ESP32
        verify_esp32_model(args.esp32_ip, args.esp32_port)
        
    elif args.action == 'package':
        # Создание пакета развертывания
        output_path = args.output or "voicex_deployment.zip"
        create_deployment_package(args.model_dir, output_path)
        
    elif args.action == 'project':
        # Создание проекта PlatformIO
        project_dir = args.output or "voicex_esp32"
        project = PlatformIOProject(project_dir)
        project.create_project()
        
        if args.cpp_file:
            project.copy_source_files(args.cpp_file)
        
        # Подготовка данных
        spiffs_dir = project.project_dir / "data"
        uploader.prepare_spiffs_data(args.model_dir, spiffs_dir)
        
        print(f"✅ Проект PlatformIO создан в {project_dir}")
        print("📝 Следующие шаги:")
        print(f"   1. cd {project_dir}")
        print("   2. pio run")
        print("   3. pio run --target upload")
        print("   4. pio run --target uploadfs")
        
    elif args.action == 'web':
        # Создание веб-интерфейса
        web_dir = args.output or "web_interface"
        create_web_interface_files(web_dir)
        
    print("=" * 40)
    print("🎉 Готово!")

if __name__ == "__main__":
    main()