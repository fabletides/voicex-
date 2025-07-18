#!/usr/bin/env python3
"""
Скрипт подготовки данных Vibravox для обучения VoiceX
Загрузка, очистка и предобработка данных
"""

import os
import shutil
import requests
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse

class VibravoxDataPreparator:
    """Класс для подготовки данных Vibravox"""
    
    def __init__(self, output_dir='./vibravox_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_vibravox_dataset(self):
        """Загрузка датасета Vibravox с GitHub"""
        print("🔄 Загрузка датасета Vibravox...")
        
        # URL репозитория
        repo_url = "https://github.com/jhauret/vibravox"
        dataset_urls = {
            'speech': 'https://github.com/jhauret/vibravox/raw/main/data/speech.zip',
            'throat': 'https://github.com/jhauret/vibravox/raw/main/data/throat_microphone.zip',
            'metadata': 'https://github.com/jhauret/vibravox/raw/main/metadata.csv'
        }
        
        # Создание директорий
        (self.output_dir / 'speech').mkdir(exist_ok=True)
        (self.output_dir / 'throat_microphone').mkdir(exist_ok=True)
        
        # Загрузка файлов
        for name, url in dataset_urls.items():
            print(f"📥 Загружаем {name}...")
            
            if name == 'metadata':
                # Загрузка метаданных
                response = requests.get(url)
                with open(self.output_dir / 'metadata.csv', 'wb') as f:
                    f.write(response.content)
            else:
                # Загрузка и распаковка архивов
                zip_path = self.output_dir / f'{name}.zip'
                
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f, tqdm(
                    desc=f"Downloading {name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                
                # Распаковка
                print(f"📦 Распаковка {name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir / name)
                
                # Удаление архива
                zip_path.unlink()
        
        print("✅ Датасет Vibravox загружен успешно!")
        
    def analyze_dataset(self):
        """Анализ структуры и качества датасета"""
        print("🔄 Анализ датасета...")
        
        # Поиск аудиофайлов
        speech_files = list((self.output_dir / 'speech').rglob('*.wav'))
        throat_files = list((self.output_dir / 'throat_microphone').rglob('*.wav'))
        
        print(f"📊 Статистика датасета:")
        print(f"   Файлы обычной речи: {len(speech_files)}")
        print(f"   Файлы горлового микрофона: {len(throat_files)}")
        
        # Анализ качества аудио
        durations = []
        sample_rates = []
        valid_pairs = 0
        
        for speech_file in tqdm(speech_files[:100], desc="Анализ качества"):  # Анализируем первые 100
            try:
                # Загрузка аудио
                y, sr = librosa.load(speech_file, sr=None)
                duration = len(y) / sr
                
                durations.append(duration)
                sample_rates.append(sr)
                
                # Проверка наличия парного файла
                relative_path = speech_file.relative_to(self.output_dir / 'speech')
                throat_file = self.output_dir / 'throat_microphone' / relative_path
                
                if throat_file.exists():
                    valid_pairs += 1
                    
            except Exception as e:
                print(f"❌ Ошибка анализа {speech_file}: {e}")
        
        print(f"📈 Анализ качества (на основе {len(durations)} файлов):")
        print(f"   Валидных пар: {valid_pairs}")
        print(f"   Средняя длительность: {np.mean(durations):.2f}с")
        print(f"   Частота дискретизации: {np.unique(sample_rates)}")
        print(f"   Диапазон длительностей: {np.min(durations):.2f}с - {np.max(durations):.2f}с")
        
        return {
            'speech_files': len(speech_files),
            'throat_files': len(throat_files),
            'valid_pairs': valid_pairs,
            'mean_duration': np.mean(durations),
            'sample_rates': np.unique(sample_rates).tolist()
        }
    
    def preprocess_audio_files(self, target_sr=22050, target_duration=3.0):
        """Предобработка аудиофайлов"""
        print("🔄 Предобработка аудиофайлов...")
        
        # Создание директорий для обработанных файлов
        processed_dir = self.output_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        (processed_dir / 'speech').mkdir(exist_ok=True)
        (processed_dir / 'throat_microphone').mkdir(exist_ok=True)
        
        # Поиск парных файлов
        speech_files = list((self.output_dir / 'speech').rglob('*.wav'))
        processed_count = 0
        
        for speech_file in tqdm(speech_files, desc="Обработка аудио"):
            try:
                # Поиск парного файла горлового микрофона
                relative_path = speech_file.relative_to(self.output_dir / 'speech')
                throat_file = self.output_dir / 'throat_microphone' / relative_path
                
                if not throat_file.exists():
                    continue
                
                # Обработка файла обычной речи
                speech_processed = self._preprocess_single_file(
                    speech_file, target_sr, target_duration
                )
                
                # Обработка файла горлового микрофона
                throat_processed = self._preprocess_single_file(
                    throat_file, target_sr, target_duration
                )
                
                if speech_processed is not None and throat_processed is not None:
                    # Сохранение обработанных файлов
                    speech_output = processed_dir / 'speech' / relative_path
                    throat_output = processed_dir / 'throat_microphone' / relative_path
                    
                    # Создание директорий
                    speech_output.parent.mkdir(parents=True, exist_ok=True)
                    throat_output.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Сохранение
                    sf.write(speech_output, speech_processed, target_sr)
                    sf.write(throat_output, throat_processed, target_sr)
                    
                    processed_count += 1
                    
            except Exception as e:
                print(f"❌ Ошибка обработки {speech_file}: {e}")
        
        print(f"✅ Обработано {processed_count} пар файлов")
        return processed_count
    
    def _preprocess_single_file(self, file_path, target_sr, target_duration):
        """Обработка одного аудиофайла"""
        try:
            # Загрузка
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # Нормализация громкости
            y = librosa.util.normalize(y)
            
            # Удаление тишины
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Приведение к целевой длительности
            target_samples = int(target_sr * target_duration)
            
            if len(y) > target_samples:
                # Обрезка
                y = y[:target_samples]
            elif len(y) < target_samples:
                # Дополнение нулями
                y = np.pad(y, (0, target_samples - len(y)), mode='constant')
            
            # Применение фильтра для улучшения качества
            y = self._apply_audio_filters(y, target_sr)
            
            return y
            
        except Exception as e:
            print(f"❌ Ошибка обработки {file_path}: {e}")
            return None
    
    def _apply_audio_filters(self, y, sr):
        """Применение аудиофильтров для улучшения качества"""
        # Высокочастотный фильтр для удаления низкочастотного шума
        y = librosa.effects.preemphasis(y)
        
        # Подавление шума (простая версия)
        # В реальном проекте здесь можно использовать более сложные алгоритмы
        y = librosa.effects.trim(y, top_db=15)[0]
        
        return y
    
    def create_speaker_splits(self):
        """Создание разделения по говорящим для обучения"""
        print("🔄 Создание разделения по говорящим...")
        
        processed_dir = self.output_dir / 'processed'
        if not processed_dir.exists():
            print("❌ Сначала запустите предобработку данных")
            return
        
        # Поиск всех обработанных файлов
        speech_files = list((processed_dir / 'speech').rglob('*.wav'))
        
        # Извлечение информации о говорящих
        speaker_info = {}
        for file_path in speech_files:
            speaker_id = self._extract_speaker_id(file_path)
            if speaker_id not in speaker_info:
                speaker_info[speaker_id] = []
            speaker_info[speaker_id].append(file_path)
        
        print(f"📊 Найдено говорящих: {len(speaker_info)}")
        for speaker, files in speaker_info.items():
            print(f"   {speaker}: {len(files)} файлов")
        
        # Создание разделения train/validation/test
        splits = {'train': [], 'validation': [], 'test': []}
        
        for speaker, files in speaker_info.items():
            n_files = len(files)
            
            # 70% - train, 15% - validation, 15% - test
            train_size = int(0.7 * n_files)
            val_size = int(0.15 * n_files)
            
            splits['train'].extend(files[:train_size])
            splits['validation'].extend(files[train_size:train_size + val_size])
            splits['test'].extend(files[train_size + val_size:])
        
        # Сохранение информации о разделении
        split_info = {}
        for split_name, files in splits.items():
            split_info[split_name] = [str(f.relative_to(processed_dir / 'speech')) for f in files]
            print(f"📁 {split_name}: {len(files)} файлов")
        
        # Сохранение в JSON
        import json
        with open(self.output_dir / 'data_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print("✅ Разделение данных создано")
        return split_info
    
    def _extract_speaker_id(self, file_path):
        """Извлечение ID говорящего из пути файла"""
        parts = str(file_path).split('/')
        for part in parts:
            if 'speaker' in part.lower() or 'spk' in part.lower():
                return part
        # Если не найдено, используем имя родительской директории
        return file_path.parent.name
    
    def generate_dataset_statistics(self):
        """Генерация статистики датасета"""
        print("🔄 Генерация статистики датасета...")
        
        processed_dir = self.output_dir / 'processed'
        if not processed_dir.exists():
            print("❌ Сначала запустите предобработку данных")
            return
        
        # Загрузка информации о разделении
        try:
            with open(self.output_dir / 'data_splits.json', 'r') as f:
                splits = json.load(f)
        except FileNotFoundError:
            print("❌ Сначала создайте разделение данных")
            return
        
        stats = {
            'total_files': 0,
            'total_duration': 0.0,
            'speakers': set(),
            'splits': {}
        }
        
        for split_name, files in splits.items():
            split_stats = {
                'files': len(files),
                'duration': 0.0,
                'speakers': set()
            }
            
            for file_rel_path in tqdm(files, desc=f"Анализ {split_name}"):
                try:
                    file_path = processed_dir / 'speech' / file_rel_path
                    y, sr = librosa.load(file_path, sr=None)
                    duration = len(y) / sr
                    
                    split_stats['duration'] += duration
                    speaker = self._extract_speaker_id(file_path)
                    split_stats['speakers'].add(speaker)
                    stats['speakers'].add(speaker)
                    
                except Exception as e:
                    print(f"❌ Ошибка анализа {file_rel_path}: {e}")
            
            split_stats['speakers'] = len(split_stats['speakers'])
            stats['splits'][split_name] = split_stats
            stats['total_files'] += split_stats['files']
            stats['total_duration'] += split_stats['duration']
        
        stats['speakers'] = len(stats['speakers'])
        
        # Сохранение статистики
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Вывод статистики
        print("📊 Статистика датасета:")
        print(f"   Общее количество файлов: {stats['total_files']}")
        print(f"   Общая продолжительность: {stats['total_duration']:.1f} сек ({stats['total_duration']/3600:.1f} часов)")
        print(f"   Количество говорящих: {stats['speakers']}")
        
        for split_name, split_stats in stats['splits'].items():
            print(f"   {split_name.capitalize()}:")
            print(f"     Файлы: {split_stats['files']}")
            print(f"     Продолжительность: {split_stats['duration']:.1f} сек")
            print(f"     Говорящих: {split_stats['speakers']}")
        
        return stats

def main():
    """Основная функция подготовки данных"""
    parser = argparse.ArgumentParser(description='Подготовка данных Vibravox для VoiceX')
    parser.add_argument('--output_dir', type=str, default='./vibravox_data',
                       help='Директория для сохранения данных')
    parser.add_argument('--download', action='store_true',
                       help='Загрузить датасет Vibravox')
    parser.add_argument('--preprocess', action='store_true',
                       help='Предобработать аудиофайлы')
    parser.add_argument('--analyze', action='store_true',
                       help='Проанализировать датасет')
    parser.add_argument('--create_splits', action='store_true',
                       help='Создать разделение на train/val/test')
    parser.add_argument('--stats', action='store_true',
                       help='Сгенерировать статистику')
    parser.add_argument('--all', action='store_true',
                       help='Выполнить все этапы подготовки')
    parser.add_argument('--target_sr', type=int, default=22050,
                       help='Целевая частота дискретизации')
    parser.add_argument('--target_duration', type=float, default=3.0,
                       help='Целевая продолжительность файлов в секундах')
    
    args = parser.parse_args()
    
    # Создание препаратора данных
    preparator = VibravoxDataPreparator(args.output_dir)
    
    print("🚀 Запуск подготовки данных Vibravox для VoiceX")
    print("=" * 50)
    
    try:
        if args.all or args.download:
            preparator.download_vibravox_dataset()
        
        if args.all or args.analyze:
            analysis_results = preparator.analyze_dataset()
        
        if args.all or args.preprocess:
            processed_count = preparator.preprocess_audio_files(
                target_sr=args.target_sr,
                target_duration=args.target_duration
            )
        
        if args.all or args.create_splits:
            split_info = preparator.create_speaker_splits()
        
        if args.all or args.stats:
            stats = preparator.generate_dataset_statistics()
        
        print("=" * 50)
        print("🎉 Подготовка данных завершена успешно!")
        print(f"📁 Данные сохранены в: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка при подготовке данных: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())