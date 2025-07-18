#!/usr/bin/env python3
"""
Основной скрипт запуска полного пайплайна обучения VoiceX
Автоматизирует весь процесс от загрузки данных до тестирования модели
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import time

class VoiceXPipelineRunner:
    """Класс для запуска полного пайплайна VoiceX"""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config['base_directory'])
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # Создание директорий
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command, description):
        """Запуск команды с логированием"""
        print(f"\n🔄 {description}")
        print(f"📝 Команда: {' '.join(command)}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {description} завершено успешно ({duration:.1f}с)")
                if result.stdout:
                    print("📤 Вывод:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ {description} завершилось с ошибкой ({duration:.1f}с)")
                print("📤 Stdout:", result.stdout)
                print("📤 Stderr:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Ошибка выполнения команды: {e}")
            return False
    
    def step_1_data_preparation(self):
        """Шаг 1: Подготовка данных"""
        print("\n" + "="*60)
        print("📥 ШАГ 1: ПОДГОТОВКА ДАННЫХ VIBRAVOX")
        print("="*60)
        
        # Команда для подготовки данных
        command = [
            sys.executable, 'data_preparation.py',
            '--output_dir', str(self.data_dir),
            '--all',
            '--target_sr', str(self.config['sample_rate']),
            '--target_duration', str(self.config['duration'])
        ]
        
        success = self.run_command(command, "Подготовка данных Vibravox")
        
        if success:
            # Проверка результатов
            stats_file = self.data_dir / 'dataset_stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                print(f"\n📊 Статистика подготовленных данных:")
                print(f"   Общее количество файлов: {stats['total_files']}")
                print(f"   Общая продолжительность: {stats['total_duration']:.1f} сек")
                print(f"   Количество говорящих: {stats['speakers']}")
        
        return success
    
    def step_2_model_training(self):
        """Шаг 2: Обучение модели"""
        print("\n" + "="*60)
        print("🧠 ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ")
        print("="*60)
        
        # Команда для обучения
        command = [
            sys.executable, 'train_voicex_model.py',
            '--data_path', str(self.data_dir / 'processed'),
            '--model_size', self.config['model_size'],
            '--epochs', str(self.config['epochs']),
            '--output_dir', str(self.models_dir)
        ]
        
        success = self.run_command(command, "Обучение модели VoiceX")
        
        if success:
            # Проверка созданных моделей
            keras_model = self.models_dir / 'voicex_model.h5'
            tflite_model = self.models_dir / 'voicex_model.tflite'
            
            if keras_model.exists() and tflite_model.exists():
                print(f"\n✅ Модели созданы:")
                print(f"   Keras: {keras_model} ({keras_model.stat().st_size / 1024:.1f} KB)")
                print(f"   TFLite: {tflite_model} ({tflite_model.stat().st_size / 1024:.1f} KB)")
        
        return success
    
    def step_3_model_testing(self):
        """Шаг 3: Тестирование модели"""
        print("\n" + "="*60)
        print("🧪 ШАГ 3: ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("="*60)
        
        # Команда для тестирования
        command = [
            sys.executable, 'model_testing.py',
            '--model_path', str(self.models_dir / 'voicex_model.h5'),
            '--config_path', str(self.models_dir / 'config.json'),
            '--scaler_path', str(self.models_dir / 'scaler.pkl'),
            '--test_data', str(self.data_dir / 'processed'),
            '--output_dir', str(self.results_dir),
            '--num_samples', str(self.config['test_samples'])
        ]
        
        success = self.run_command(command, "Тестирование модели")
        
        if success:
            # Проверка результатов тестирования
            metrics_file = self.results_dir / 'test_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                print(f"\n📊 Результаты тестирования:")
                print(f"   Успешных конвертаций: {metrics['successful_conversions']}/{metrics['total_samples']}")
                print(f"   Процент успеха: {metrics['success_rate']*100:.1f}%")
                
                if 'mse' in metrics:
                    print(f"   MSE: {metrics['mse']:.6f}")
                    print(f"   MAE: {metrics['mae']:.6f}")
                    print(f"   Косинусная схожесть: {metrics['cosine_similarity']:.3f}")
        
        return success
    
    def step_4_create_demo(self):
        """Шаг 4: Создание демо"""
        print("\n" + "="*60)
        print("🎬 ШАГ 4: СОЗДАНИЕ ДЕМО")
        print("="*60)
        
        # Поиск примеров для демо
        processed_dir = self.data_dir / 'processed'
        throat_files = list((processed_dir / 'throat_microphone').rglob('*.wav'))
        
        if not throat_files:
            print("❌ Не найдены файлы для создания демо")
            return False
        
        # Выбор первого доступного файла
        throat_file = throat_files[0]
        relative_path = throat_file.relative_to(processed_dir / 'throat_microphone')
        speech_file = processed_dir / 'speech' / relative_path
        
        if not speech_file.exists():
            print("❌ Не найден парный файл обычной речи для демо")
            return False
        
        # Команда для создания демо
        command = [
            sys.executable, 'model_testing.py',
            '--model_path', str(self.models_dir / 'voicex_model.h5'),
            '--config_path', str(self.models_dir / 'config.json'),
            '--scaler_path', str(self.models_dir / 'scaler.pkl'),
            '--throat_file', str(throat_file),
            '--speech_file', str(speech_file),
            '--output_dir', str(self.results_dir),
            '--create_demo'
        ]
        
        success = self.run_command(command, "Создание демо-сравнения")
        
        if success:
            demo_file = self.results_dir / 'demo' / 'demo.html'
            if demo_file.exists():
                print(f"\n🎉 Демо создано!")
                print(f"   Откройте в браузере: {demo_file}")
        
        return success
    
    def step_5_esp32_preparation(self):
        """Шаг 5: Подготовка для ESP32"""
        print("\n" + "="*60)
        print("🔧 ШАГ 5: ПОДГОТОВКА ДЛЯ ESP32")
        print("="*60)
        
        # Создание директории для ESP32
        esp32_dir = self.base_dir / 'esp32_deployment'
        esp32_dir.mkdir(exist_ok=True)
        
        # Копирование TFLite модели
        tflite_model = self.models_dir / 'voicex_model.tflite'
        if tflite_model.exists():
            import shutil
            shutil.copy2(tflite_model, esp32_dir / 'voice_conversion_model.tflite')
            print(f"✅ TFLite модель скопирована для ESP32")
        
        # Создание конфигурационного файла для ESP32
        esp32_config = {
            'model_file': 'voice_conversion_model.tflite',
            'sample_rate': self.config['sample_rate'],
            'duration': self.config['duration'],
            'input_dim': self.config.get('input_dim', 1024),
            'output_dim': self.config.get('output_dim', 1024),
            'model_size_kb': tflite_model.stat().st_size / 1024 if tflite_model.exists() else 0
        }
        
        with open(esp32_dir / 'esp32_config.json', 'w') as f:
            json.dump(esp32_config, f, indent=2)
        
        # Создание README для ESP32
        readme_content = """# VoiceX ESP32 Deployment

## Файлы для загрузки на ESP32:

1. `voice_conversion_model.tflite` - Обученная модель TensorFlow Lite
2. `esp32_config.json` - Конфигурация модели

## Инструкции по использованию:

1. Скопируйте `voice_conversion_model.tflite` в SPIFFS ESP32
2. Обновите конфигурацию в коде ESP32 согласно `esp32_config.json`
3. Загрузите прошивку VoiceX на ESP32

## Характеристики модели:

- Размер модели: {:.1f} KB
- Частота дискретизации: {} Гц
- Длительность обработки: {} сек
- Входная размерность: {}
- Выходная размерность: {}

""".format(
            esp32_config['model_size_kb'],
            esp32_config['sample_rate'],
            esp32_config['duration'],
            esp32_config['input_dim'],
            esp32_config['output_dim']
        )
        
        with open(esp32_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"📁 Файлы для ESP32 подготовлены в: {esp32_dir}")
        
        return True
    
    def run_full_pipeline(self):
        """Запуск полного пайплайна"""
        print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА VOICEX")
        print("🎯 Цель: Обучение модели преобразования голоса для ESP32")
        print("📊 Датасет: Vibravox")
        print("=" * 80)
        
        start_time = time.time()
        
        # Этапы пайплайна
        steps = [
            ("1️⃣", self.step_1_data_preparation),
            ("2️⃣", self.step_2_model_training),
            ("3️⃣", self.step_3_model_testing),
            ("4️⃣", self.step_4_create_demo),
            ("5️⃣", self.step_5_esp32_preparation)
        ]
        
        successful_steps = 0
        
        for step_icon, step_func in steps:
            print(f"\n{step_icon} Выполняется: {step_func.__name__}")
            
            if step_func():
                successful_steps += 1
                print(f"✅ {step_func.__name__} - УСПЕШНО")
            else:
                print(f"❌ {step_func.__name__} - ОШИБКА")
                
                # Спросить у пользователя, продолжать ли
                user_input = input("\n❓ Продолжить выполнение несмотря на ошибку? (y/N): ")
                if user_input.lower() != 'y':
                    print("⏹️ Выполнение пайплайна остановлено пользователем")
                    break
        
        total_time = time.time() - start_time
        
        # Итоговый отчет
        print("\n" + "="*80)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("="*80)
        print(f"⏱️  Общее время выполнения: {total_time/60:.1f} минут")
        print(f"✅ Успешных этапов: {successful_steps}/{len(steps)}")
        
        if successful_steps == len(steps):
            print("\n🎉 ПАЙПЛАЙН ВЫПОЛНЕН ПОЛНОСТЬЮ!")
            print("🚀 Модель VoiceX готова для развертывания на ESP32!")
            
            # Путь к результатам
            print(f"\n📁 Результаты:")
            print(f"   Модели: {self.models_dir}")
            print(f"   Тестирование: {self.results_dir}")
            print(f"   ESP32 файлы: {self.base_dir}/esp32_deployment")
            
            # Демо
            demo_file = self.results_dir / 'demo' / 'demo.html'
            if demo_file.exists():
                print(f"   Демо: {demo_file}")
        else:
            print(f"\n⚠️  Пайплайн выполнен частично ({successful_steps}/{len(steps)} этапов)")
            print("🔧 Проверьте ошибки и запустите заново")
        
        return successful_steps == len(steps)

def load_config(config_path):
    """Загрузка конфигурации пайплайна"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Конфигурация по умолчанию
    return {
        'base_directory': './voicex_pipeline',
        'sample_rate': 22050,
        'duration': 3.0,
        'model_size': 'small',
        'epochs': 50,
        'test_samples': 20
    }

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Полный пайплайн обучения VoiceX')
    parser.add_argument('--config', type=str, 
                       help='Путь к файлу конфигурации JSON')
    parser.add_argument('--base_dir', type=str, default='./voicex_pipeline',
                       help='Базовая директория для всех файлов')
    parser.add_argument('--skip_data', action='store_true',
                       help='Пропустить подготовку данных')
    parser.add_argument('--skip_training', action='store_true',
                       help='Пропустить обучение модели')
    parser.add_argument('--skip_testing', action='store_true',
                       help='Пропустить тестирование')
    parser.add_argument('--only_demo', action='store_true',
                       help='Создать только демо (требует готовой модели)')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    config['base_directory'] = args.base_dir
    
    # Создание runner'а
    runner = VoiceXPipelineRunner(config)
    
    # Сохранение конфигурации
    config_file = Path(config['base_directory']) / 'pipeline_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Конфигурация сохранена: {config_file}")
    
    try:
        if args.only_demo:
            # Только создание демо
            success = runner.step_4_create_demo()
            return 0 if success else 1
        else:
            # Полный пайплайн
            success = runner.run_full_pipeline()
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Выполнение прервано пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())