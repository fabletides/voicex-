#!/usr/bin/env python3
"""
VoiceX ML Training Pipeline для Vibravox Dataset
Обучение моделей преобразования голоса для устройства VoiceX
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Конфигурация
CONFIG = {
    'sample_rate': 22050,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 80,
    'duration': 3.0,  # секунды
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'model_size': 'small',  # small, medium, large
}

class VibravoxDataProcessor:
    """Обработчик данных Vibravox для обучения моделей VoiceX"""
    
    def __init__(self, data_path, config=CONFIG):
        self.data_path = Path(data_path)
        self.config = config
        self.scaler = StandardScaler()
        
    def load_vibravox_data(self):
        """Загрузка и обработка данных Vibravox"""
        print("🔄 Загрузка данных Vibravox...")
        
        # Поиск аудиофайлов
        audio_files = []
        metadata = []
        
        # Структура Vibravox: speech/, throat_microphone/, etc.
        speech_dir = self.data_path / "speech"
        throat_dir = self.data_path / "throat_microphone"
        
        if not speech_dir.exists() or not throat_dir.exists():
            raise FileNotFoundError(f"Не найдены директории speech/ и throat_microphone/ в {self.data_path}")
        
        # Парсинг аудиофайлов
        for speech_file in speech_dir.glob("**/*.wav"):
            # Поиск соответствующего файла горлового микрофона
            relative_path = speech_file.relative_to(speech_dir)
            throat_file = throat_dir / relative_path
            
            if throat_file.exists():
                audio_files.append({
                    'speech_path': str(speech_file),
                    'throat_path': str(throat_file),
                    'speaker_id': self._extract_speaker_id(speech_file),
                    'session_id': self._extract_session_id(speech_file),
                    'utterance_id': speech_file.stem
                })
        
        print(f"✅ Найдено {len(audio_files)} пар аудиофайлов")
        return audio_files
    
    def _extract_speaker_id(self, file_path):
        """Извлечение ID говорящего из пути файла"""
        parts = str(file_path).split('/')
        for part in parts:
            if part.startswith('speaker_') or part.startswith('spk'):
                return part
        return "unknown"
    
    def _extract_session_id(self, file_path):
        """Извлечение ID сессии из пути файла"""
        parts = str(file_path).split('/')
        for part in parts:
            if part.startswith('session_') or part.startswith('sess'):
                return part
        return "unknown"
    
    def extract_audio_features(self, audio_path, is_throat=False):
        """Извлечение признаков из аудио"""
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            
            # Обрезка/дополнение до нужной длительности
            target_length = int(self.config['sample_rate'] * self.config['duration'])
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
            # Извлечение признаков
            features = {}
            
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length'],
                n_mels=self.config['n_mels']
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
            
            # MFCC
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=13,
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            features['mfcc'] = mfcc
            
            # Pitch (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            features['f0'] = np.nan_to_num(f0)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            features['spectral_centroid'] = spectral_centroids
            features['spectral_rolloff'] = spectral_rolloff
            features['zcr'] = zero_crossing_rate
            
            # Дополнительные признаки для горлового микрофона
            if is_throat:
                # Специфические признаки для throat microphone
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma'] = chroma
                
                # Энергия в разных частотных полосах
                features['energy'] = librosa.feature.rms(y=y)[0]
            
            return features
            
        except Exception as e:
            print(f"❌ Ошибка обработки {audio_path}: {e}")
            return None
    
    def create_training_dataset(self, audio_files, test_size=0.2):
        """Создание тренировочного датасета"""
        print("🔄 Создание тренировочного датасета...")
        
        X_speech = []  # Признаки обычной речи
        X_throat = []  # Признаки горлового микрофона
        y_speech = []  # Целевые признаки (обычная речь)
        
        valid_samples = 0
        
        for file_info in tqdm(audio_files, desc="Обработка аудиофайлов"):
            # Обработка обычной речи (target)
            speech_features = self.extract_audio_features(file_info['speech_path'], is_throat=False)
            
            # Обработка горлового микрофона (input)
            throat_features = self.extract_audio_features(file_info['throat_path'], is_throat=True)
            
            if speech_features is not None and throat_features is not None:
                # Используем mel-spectrogram как основные признаки
                X_throat.append(throat_features['mel_spectrogram'].flatten())
                y_speech.append(speech_features['mel_spectrogram'].flatten())
                
                # Добавляем дополнительные признаки
                additional_throat = np.concatenate([
                    throat_features['mfcc'].flatten(),
                    throat_features['f0'][:len(throat_features['spectral_centroid'])],
                    throat_features['spectral_centroid'],
                    throat_features['zcr']
                ])
                
                additional_speech = np.concatenate([
                    speech_features['mfcc'].flatten(),
                    speech_features['f0'][:len(speech_features['spectral_centroid'])],
                    speech_features['spectral_centroid'],
                    speech_features['zcr']
                ])
                
                X_speech.append(additional_throat)
                valid_samples += 1
        
        print(f"✅ Обработано {valid_samples} валидных образцов")
        
        # Конвертация в numpy arrays
        X_throat = np.array(X_throat)
        y_speech = np.array(y_speech)
        
        # Нормализация
        X_throat_scaled = self.scaler.fit_transform(X_throat)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_throat_scaled, y_speech, 
            test_size=test_size, 
            random_state=42,
            stratify=None
        )
        
        print(f"📊 Размеры датасета:")
        print(f"   Train: X={X_train.shape}, y={y_train.shape}")
        print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
        
        return (X_train, X_test, y_train, y_test)

class VoiceXModel:
    """Модель преобразования голоса для VoiceX"""
    
    def __init__(self, input_dim, output_dim, model_size='small'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_size = model_size
        self.model = None
        
    def build_model(self):
        """Построение модели нейронной сети"""
        print(f"🔄 Построение модели размера '{self.model_size}'...")
        
        # Конфигурация в зависимости от размера модели
        if self.model_size == 'small':
            hidden_layers = [256, 128, 64]
            dropout_rate = 0.3
        elif self.model_size == 'medium':
            hidden_layers = [512, 256, 128, 64]
            dropout_rate = 0.4
        else:  # large
            hidden_layers = [1024, 512, 256, 128, 64]
            dropout_rate = 0.5
        
        # Архитектура модели
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_dim,)),
            keras.layers.BatchNormalization(),
        ])
        
        # Скрытые слои
        for i, units in enumerate(hidden_layers):
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Выходной слой
        model.add(keras.layers.Dense(self.output_dim, activation='linear'))
        
        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print(f"✅ Модель построена. Параметров: {model.count_params():,}")
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100):
        """Обучение модели"""
        print("🔄 Начинаем обучение модели...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=10,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_voicex_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            batch_size=CONFIG['batch_size'],
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Обучение завершено!")
        return history
    
    def convert_to_tflite(self, output_path='voicex_model.tflite'):
        """Конвертация модели в TensorFlow Lite для ESP32"""
        print("🔄 Конвертация в TensorFlow Lite...")
        
        # Конвертер
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Оптимизация для микроконтроллеров
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        # Квантизация для уменьшения размера
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Конвертация
        tflite_model = converter.convert()
        
        # Сохранение
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Модель сохранена: {output_path} ({len(tflite_model)/1024:.1f} KB)")
        return tflite_model
    
    def _representative_dataset(self):
        """Репрезентативный датасет для квантизации"""
        # Здесь должны быть примеры данных для квантизации
        for _ in range(100):
            yield [np.random.float32(np.random.random((1, self.input_dim)))]

def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test, scaler):
    """Оценка качества модели"""
    print("🔄 Оценка модели...")
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"📊 Метрики модели:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    
    # Сравнение предсказаний и истинных значений
    sample_idx = np.random.choice(len(y_test), 5)
    
    for i, idx in enumerate(sample_idx):
        plt.subplot(2, 3, i+1)
        plt.plot(y_test[idx][:100], label='True', alpha=0.7)
        plt.plot(y_pred[idx][:100], label='Predicted', alpha=0.7)
        plt.title(f'Sample {idx}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'mse': mse, 'mae': mae}

def main():
    """Основная функция обучения"""
    parser = argparse.ArgumentParser(description='VoiceX ML Training Pipeline')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Путь к датасету Vibravox')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Размер модели')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Директория для сохранения моделей')
    
    args = parser.parse_args()
    
    # Создание директории для моделей
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Запуск VoiceX ML Training Pipeline")
    print("=" * 50)
    
    # 1. Загрузка и обработка данных
    processor = VibravoxDataProcessor(args.data_path, CONFIG)
    audio_files = processor.load_vibravox_data()
    
    # 2. Создание датасета
    X_train, X_test, y_train, y_test = processor.create_training_dataset(audio_files)
    
    # 3. Построение модели
    model_builder = VoiceXModel(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        model_size=args.model_size
    )
    model = model_builder.build_model()
    
    # 4. Обучение
    history = model_builder.train(X_train, y_train, X_test, y_test, epochs=args.epochs)
    
    # 5. Визуализация
    plot_training_history(history)
    
    # 6. Оценка
    metrics = evaluate_model(model, X_test, y_test, processor.scaler)
    
    # 7. Сохранение моделей
    # Keras модель
    keras_path = os.path.join(args.output_dir, 'voicex_model.h5')
    model.save(keras_path)
    print(f"✅ Keras модель сохранена: {keras_path}")
    
    # TensorFlow Lite модель
    tflite_path = os.path.join(args.output_dir, 'voicex_model.tflite')
    model_builder.convert_to_tflite(tflite_path)
    
    # Сохранение скейлера
    import joblib
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    joblib.dump(processor.scaler, scaler_path)
    print(f"✅ Скейлер сохранен: {scaler_path}")
    
    # Сохранение конфигурации
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'CONFIG': CONFIG,
            'model_size': args.model_size,
            'input_dim': X_train.shape[1],
            'output_dim': y_train.shape[1],
            'metrics': metrics
        }, f, indent=2)
    
    print("=" * 50)
    print("🎉 Обучение завершено успешно!")
    print(f"📁 Модели сохранены в: {args.output_dir}")
    print(f"📊 Финальные метрики: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")

if __name__ == "__main__":
    main()