#!/usr/bin/env python3
"""
Скрипт тестирования обученной модели VoiceX
Оценка качества преобразования голоса и создание демо
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

class VoiceXModelTester:
    """Класс для тестирования модели VoiceX"""
    
    def __init__(self, model_path, config_path, scaler_path):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.scaler_path = Path(scaler_path)
        
        # Загрузка конфигурации
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Загрузка модели
        if str(model_path).endswith('.tflite'):
            self.model = self._load_tflite_model()
            self.model_type = 'tflite'
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.model_type = 'keras'
        
        # Загрузка скейлера
        self.scaler = joblib.load(scaler_path)
        
        # Инициализация параметров обработки аудио
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.duration = self.config.get('duration', 3.0)
        self.n_fft = self.config.get('n_fft', 2048)
        self.hop_length = self.config.get('hop_length', 512)
        self.n_mels = self.config.get('n_mels', 128)
        self.n_mfcc = self.config.get('n_mfcc', 13)
        
        print(f"✅ Модель загружена: {model_path}")
        print(f"📊 Тип модели: {self.model_type}")
        print(f"🔧 Входная размерность: {self.config.get('input_dim', 'не указана')}")
        print(f"🎯 Выходная размерность: {self.config.get('output_dim', 'не указана')}")
    
    def _load_tflite_model(self):
        """Загрузка TensorFlow Lite модели"""
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return interpreter
    
    def preprocess_throat_signal(self, audio_signal, sr):
        """
        Предварительная обработка сигнала с горлового микрофона
        Включает фильтрацию шума и усиление полезного сигнала
        """
        # Нормализация амплитуды
        audio_signal = librosa.util.normalize(audio_signal)
        
        # Применение полосового фильтра для голосовых частот (80-4000 Hz)
        from scipy import signal
        nyquist = sr / 2
        low = 80 / nyquist
        high = 4000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, audio_signal)
        
        # Подавление шума через спектральное вычитание
        stft = librosa.stft(filtered_signal, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Оценка шума из первых и последних 10% сигнала
        noise_frames = int(0.1 * magnitude.shape[1])
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        noise_profile = np.maximum(noise_profile, 
                                 np.mean(magnitude[:, -noise_frames:], axis=1, keepdims=True))
        
        # Спектральное вычитание с коэффициентом подавления
        alpha = 2.0  # коэффициент подавления шума
        magnitude_clean = magnitude - alpha * noise_profile
        magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)
        
        # Восстановление сигнала
        stft_clean = magnitude_clean * np.exp(1j * phase)
        cleaned_signal = librosa.istft(stft_clean, hop_length=self.hop_length)
        
        return cleaned_signal
    
    def extract_features_from_audio(self, audio_path, is_throat=False):
        """Извлечение признаков из аудиофайла"""
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Предварительная обработка для сигналов горлового микрофона
            if is_throat:
                y = self.preprocess_throat_signal(y, sr)
            
            # Обрезка/дополнение до нужной длительности
            target_length = int(self.sample_rate * self.duration)
            if len(y) > target_length:
                # Берем центральную часть
                start = (len(y) - target_length) // 2
                y = y[start:start + target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
            # Извлечение признаков
            features = []
            
            # 1. Mel-spectrogram (основной признак для голоса)
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=80,  # минимальная частота для голоса
                fmax=8000  # максимальная частота
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.extend(mel_spec_db.flatten())
            
            # 2. MFCC (кепстральные коэффициенты)
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features.extend(mfcc.flatten())
            
            # 3. Хроматические признаки
            chroma = librosa.feature.chroma_stft(
                y=y, 
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features.extend(chroma.flatten())
            
            # 4. Спектральные признаки
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            features.extend(spectral_centroids)
            features.extend(spectral_rolloff)
            features.extend(spectral_bandwidth)
            features.extend(zero_crossing_rate)
            
            # 5. Pitch (F0) - основная частота
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            
            # Синхронизация длины f0 с другими признаками
            target_f0_length = len(spectral_centroids)
            if len(f0) > target_f0_length:
                f0 = f0[:target_f0_length]
            else:
                f0 = np.pad(f0, (0, target_f0_length - len(f0)), mode='constant')
            
            features.extend(f0)
            features.extend(voiced_probs[:len(f0)] if voiced_probs is not None else np.zeros(len(f0)))
            
            # 6. Дополнительные признаки для персонализации голоса
            if is_throat:
                # Для горлового микрофона добавляем специфические признаки
                # Энергия в различных частотных полосах
                freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
                stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
                magnitude = np.abs(stft)
                
                # Энергия в низких частотах (80-300 Hz) - фундаментальные частоты
                low_freq_mask = (freqs >= 80) & (freqs <= 300)
                low_freq_energy = np.mean(magnitude[low_freq_mask, :], axis=0)
                features.extend(low_freq_energy)
                
                # Энергия в средних частотах (300-2000 Hz) - формантные частоты
                mid_freq_mask = (freqs >= 300) & (freqs <= 2000)
                mid_freq_energy = np.mean(magnitude[mid_freq_mask, :], axis=0)
                features.extend(mid_freq_energy)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Ошибка извлечения признаков из {audio_path}: {e}")
            return None
    
    def predict_voice_conversion(self, throat_features):
        """Предсказание преобразования голоса"""
        try:
            # Проверка размерности входных данных
            expected_input_dim = self.config.get('input_dim')
            if expected_input_dim and len(throat_features) != expected_input_dim:
                print(f"⚠️ Предупреждение: размерность признаков {len(throat_features)} != ожидаемой {expected_input_dim}")
                # Обрезка или дополнение до нужного размера
                if len(throat_features) > expected_input_dim:
                    throat_features = throat_features[:expected_input_dim]
                else:
                    throat_features = np.pad(throat_features, (0, expected_input_dim - len(throat_features)), mode='constant')
            
            # Нормализация признаков
            throat_features_scaled = self.scaler.transform(throat_features.reshape(1, -1))
            
            if self.model_type == 'keras':
                # Keras модель
                prediction = self.model.predict(throat_features_scaled, verbose=0)
            else:
                # TensorFlow Lite модель
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                
                self.model.set_tensor(input_details[0]['index'], throat_features_scaled.astype(np.float32))
                self.model.invoke()
                prediction = self.model.get_tensor(output_details[0]['index'])
            
            return prediction[0]
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return None
    
    def convert_features_to_audio(self, features, output_path, target_sr=22050):
        """Конвертация признаков обратно в аудио"""
        try:
            # Разделение признаков на компоненты
            n_frames = int(self.duration * target_sr / self.hop_length)
            mel_spec_size = self.n_mels * n_frames
            
            if len(features) < mel_spec_size:
                print(f"❌ Ошибка: недостаточно признаков для реконструкции. Получено: {len(features)}, нужно минимум: {mel_spec_size}")
                return False
            
            # Извлечение mel-spectrogram
            mel_spec_flat = features[:mel_spec_size]
            mel_spec = mel_spec_flat.reshape(self.n_mels, n_frames)
            
            # Конвертация из dB в power
            mel_spec_power = librosa.db_to_power(mel_spec)
            
            # Реконструкция аудио через Griffin-Lim с улучшенными параметрами
            audio = librosa.feature.inverse.mel_to_audio(
                mel_spec_power,
                sr=target_sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                fmin=80,
                fmax=8000,
                n_iter=64  # больше итераций для лучшего качества
            )
            
            # Постобработка
            # Применение оконной функции для сглаживания
            window_size = int(0.02 * target_sr)  # 20мс окно
            if len(audio) > window_size:
                fade_in = np.linspace(0, 1, window_size)
                fade_out = np.linspace(1, 0, window_size)
                audio[:window_size] *= fade_in
                audio[-window_size:] *= fade_out
            
            # Нормализация с ограничением пиков
            audio = librosa.util.normalize(audio) * 0.8
            
            # Сохранение
            sf.write(output_path, audio, target_sr)
            return True
            
        except Exception as e:
            print(f"❌ Ошибка конвертации в аудио: {e}")
            return False
    
    def calculate_voice_quality_metrics(self, predicted_features, target_features):
        """Вычисление метрик качества голоса"""
        # Приведение к одинаковому размеру
        min_size = min(len(predicted_features), len(target_features))
        pred_trimmed = predicted_features[:min_size]
        target_trimmed = target_features[:min_size]
        
        # Основные метрики
        mse = mean_squared_error(target_trimmed, pred_trimmed)
        mae = mean_absolute_error(target_trimmed, pred_trimmed)
        cosine_sim = 1 - cosine(target_trimmed, pred_trimmed)
        
        # Специфические метрики для голоса
        # Корреляция Пирсона
        correlation = np.corrcoef(pred_trimmed, target_trimmed)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Спектральная дистанция
        spectral_distance = np.sqrt(np.mean((pred_trimmed - target_trimmed) ** 2))
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(target_trimmed ** 2)
        noise_power = np.mean((pred_trimmed - target_trimmed) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'cosine_similarity': float(cosine_sim),
            'correlation': float(correlation),
            'spectral_distance': float(spectral_distance),
            'snr_db': float(snr)
        }
    
    def test_single_sample(self, throat_audio_path, target_audio_path=None):
        """Тестирование на одном образце"""
        print(f"🔄 Тестирование: {throat_audio_path}")
        
        # Извлечение признаков из горлового микрофона
        throat_features = self.extract_features_from_audio(throat_audio_path, is_throat=True)
        if throat_features is None:
            return None
        
        # Предсказание
        predicted_features = self.predict_voice_conversion(throat_features)
        if predicted_features is None:
            return None
        
        # Сохранение результата
        output_path = Path(throat_audio_path).parent / f"converted_{Path(throat_audio_path).stem}.wav"
        success = self.convert_features_to_audio(predicted_features, output_path)
        
        result = {
            'input_file': str(throat_audio_path),
            'output_file': str(output_path) if success else None,
            'conversion_success': success,
            'throat_features_shape': throat_features.shape,
            'predicted_features_shape': predicted_features.shape
        }
        
        # Если есть целевой файл, вычисляем метрики
        if target_audio_path and Path(target_audio_path).exists():
            target_features = self.extract_features_from_audio(target_audio_path, is_throat=False)
            if target_features is not None:
                metrics = self.calculate_voice_quality_metrics(predicted_features, target_features)
                result.update(metrics)
                result['target_file'] = str(target_audio_path)
        
        return result
    
    def test_dataset(self, data_path, output_dir, num_samples=50):
        """Тестирование на датасете"""
        print(f"🔄 Тестирование на датасете: {data_path}")
        
        data_path = Path(data_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Поиск файлов горлового микрофона
        throat_files = list(data_path.rglob('*throat*.wav')) + list(data_path.rglob('*vibro*.wav'))
        
        if len(throat_files) == 0:
            # Попробуем найти в подпапках
            throat_files = list((data_path / 'throat_microphone').rglob('*.wav')) if (data_path / 'throat_microphone').exists() else []
            
        if len(throat_files) == 0:
            print("❌ Не найдены файлы горлового микрофона")
            print("💡 Попробуйте организовать данные в структуру:")
            print("   data_path/")
            print("   ├── throat_microphone/")
            print("   │   └── *.wav")
            print("   └── speech/")
            print("       └── *.wav")
            return None
        
        # Ограничение количества тестов
        throat_files = throat_files[:num_samples]
        
        results = []
        successful_conversions = 0
        
        for throat_file in tqdm(throat_files, desc="Тестирование"):
            # Поиск соответствующего файла обычной речи
            relative_path = throat_file.name
            speech_file = None
            
            # Ищем в разных возможных местах
            speech_dirs = [data_path / 'speech', data_path / 'normal', data_path / 'target']
            for speech_dir in speech_dirs:
                if speech_dir.exists():
                    potential_speech = speech_dir / relative_path
                    if potential_speech.exists():
                        speech_file = potential_speech
                        break
            
            target_file = str(speech_file) if speech_file else None
            
            # Тестирование
            result = self.test_single_sample(str(throat_file), target_file)
            
            if result:
                results.append(result)
                if result['conversion_success']:
                    successful_conversions += 1
                    
                    # Копирование результата в выходную директорию
                    if result['output_file']:
                        output_file = output_dir / f"sample_{len(results):03d}_converted.wav"
                        try:
                            import shutil
                            shutil.copy2(result['output_file'], output_file)
                            result['demo_file'] = str(output_file)
                        except Exception as e:
                            print(f"❌ Ошибка копирования: {e}")
        
        # Сохранение результатов
        results_path = output_dir / 'test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        # Вычисление общих метрик
        metrics_results = self._calculate_overall_metrics(results)
        
        # Сохранение метрик
        metrics_path = output_dir / 'test_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, indent=2, ensure_ascii=False)
        
        # Создание отчета
        self._create_test_report(output_dir, results, metrics_results)
        
        print(f"📊 Результаты тестирования:")
        print(f"   Всего тестов: {len(results)}")
        print(f"   Успешных конвертаций: {successful_conversions}")
        print(f"   Процент успеха: {successful_conversions/len(results)*100:.1f}%")
        
        if 'mse' in metrics_results:
            print(f"   Средний MSE: {metrics_results['mse']:.6f}")
            print(f"   Средний MAE: {metrics_results['mae']:.6f}")
            print(f"   Средняя косинусная схожесть: {metrics_results['cosine_similarity']:.3f}")
            print(f"   Средний SNR: {metrics_results['snr_db']:.2f} dB")
        
        return results
    
    def _calculate_overall_metrics(self, results):
        """Вычисление общих метрик"""
        metrics = {}
        
        # Фильтрация результатов с метриками
        results_with_metrics = [r for r in results if 'mse' in r]
        
        if results_with_metrics:
            metric_keys = ['mse', 'mae', 'cosine_similarity', 'correlation', 'spectral_distance', 'snr_db']
            
            for key in metric_keys:
                values = [r[key] for r in results_with_metrics if key in r]
                if values:
                    metrics[key] = np.mean(values)
                    metrics[f'{key}_std'] = np.std(values)
                    metrics[f'{key}_min'] = np.min(values)
                    metrics[f'{key}_max'] = np.max(values)
        
        metrics['total_samples'] = len(results)
        metrics['successful_conversions'] = len([r for r in results if r['conversion_success']])
        metrics['success_rate'] = metrics['successful_conversions'] / metrics['total_samples'] if metrics['total_samples'] > 0 else 0
        
        return metrics
    
    def _create_test_report(self, output_dir, results, metrics):
        """Создание HTML отчета о тестировании"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VoiceX Testing Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .samples {{ margin-top: 30px; }}
                .sample {{ background-color: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #007bff; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎙️ VoiceX Model Testing Report</h1>
                    <p>Отчет о тестировании модели персонализации голоса</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Общая статистика</h3>
                        <div class="metric-value success">{metrics['total_samples']}</div>
                        <p>Всего протестировано образцов</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Успешность конвертации</h3>
                        <div class="metric-value {'success' if metrics['success_rate'] > 0.8 else 'warning' if metrics['success_rate'] > 0.5 else 'error'}">{metrics['success_rate']*100:.1f}%</div>
                        <p>{metrics['successful_conversions']} из {metrics['total_samples']} успешных</p>
                    </div>
        """
        
        if 'mse' in metrics:
            html_content += f"""
                    <div class="metric-card">
                        <h3>Качество (MSE)</h3>
                        <div class="metric-value">{metrics['mse']:.6f}</div>
                        <p>Среднеквадратичная ошибка</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Схожесть (Cosine)</h3>
                        <div class="metric-value success">{metrics['cosine_similarity']:.3f}</div>
                        <p>Косинусная схожесть</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>SNR</h3>
                        <div class="metric-value">{metrics['snr_db']:.2f} dB</div>
                        <p>Отношение сигнал/шум</p>
                    </div>
            """
        
        html_content += """
                </div>
                
                <h2>📈 Детальные метрики</h2>
                <table>
                    <tr>
                        <th>Метрика</th>
                        <th>Среднее</th>
                        <th>Стд. откл.</th>
                        <th>Минимум</th>
                        <th>Максимум</th>
                    </tr>
        """
        
        if 'mse' in metrics:
            metric_names = {
                'mse': 'MSE',
                'mae': 'MAE', 
                'cosine_similarity': 'Cosine Similarity',
                'correlation': 'Correlation',
                'snr_db': 'SNR (dB)'
            }
            
            for key, name in metric_names.items():
                if key in metrics:
                    html_content += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{metrics[key]:.6f}</td>
                        <td>{metrics.get(f'{key}_std', 0):.6f}</td>
                        <td>{metrics.get(f'{key}_min', 0):.6f}</td>
                        <td>{metrics.get(f'{key}_max', 0):.6f}</td>
                    </tr>
                    """
        
        html_content += """
                </table>
                
                <h2>🔬 Интерпретация результатов</h2>
                <div style="background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3>Как читать метрики:</h3>
                    <ul>
                        <li><strong>MSE (Mean Squared Error):</strong> Чем меньше, тем лучше. Хорошие значения < 0.01</li>
                        <li><strong>Cosine Similarity:</strong> Чем ближе к 1, тем лучше. Хорошие значения > 0.7</li>
                        <li><strong>SNR (Signal-to-Noise Ratio):</strong> Чем больше, тем лучше. Хорошие значения > 10 dB</li>
                        <li><strong>Correlation:</strong> Чем ближе к 1, тем лучше. Хорошие значения > 0.6</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background-color: #f0f8ff; border-radius: 8px;">
                    <h3>💡 О модели VoiceX:</h3>
                    <p>VoiceX использует машинное обучение для персонализации голоса, преобразуя вибрации горлового микрофона в естественную речь. Модель обучена восстанавливать индивидуальные характеристики голоса пациента.</p>
                    <p><strong>Процесс:</strong> Горловые вибрации → Очистка сигнала → ML персонализация → Естественный голос</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = output_dir / 'test_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML отчет создан: {report_path}")
    
    def create_demo_comparison(self, throat_file, speech_file, output_dir):
        """Создание демо-сравнения оригинал vs конвертированный"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🔄 Создание демо-сравнения...")
        
        # Конвертация
        result = self.test_single_sample(throat_file, speech_file)
        
        if not result or not result['conversion_success']:
            print("❌ Не удалось создать конвертированную версию")
            return None
        
        try:
            # Загрузка аудиофайлов для визуализации
            throat_audio, sr1 = librosa.load(throat_file, sr=22050)
            converted_audio, sr3 = librosa.load(result['output_file'], sr=22050)
            
            speech_audio = None
            if speech_file and Path(speech_file).exists():
                speech_audio, sr2 = librosa.load(speech_file, sr=22050)
            
            # Создание визуализации
            num_plots = 3 if speech_audio is not None else 2
            fig, axes = plt.subplots(num_plots, 2, figsize=(15, 5*num_plots))
            
            if num_plots == 2:
                axes = axes.reshape(2, 2)
            
            # Временные сигналы
            time_samples = min(len(throat_audio), sr1*3)  # Первые 3 секунды
            axes[0, 0].plot(np.linspace(0, time_samples/sr1, time_samples), throat_audio[:time_samples])
            axes[0, 0].set_title('Throat Microphone Signal')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            time_samples_conv = min(len(converted_audio), sr3*3)
            axes[1, 0].plot(np.linspace(0, time_samples_conv/sr3, time_samples_conv), converted_audio[:time_samples_conv])
            axes[1, 0].set_title('Converted Speech Signal (VoiceX Output)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Amplitude')
            axes[1, 0].grid(True, alpha=0.3)
            
            if speech_audio is not None:
                time_samples_orig = min(len(speech_audio), sr2*3)
                axes[2, 0].plot(np.linspace(0, time_samples_orig/sr2, time_samples_orig), speech_audio[:time_samples_orig])
                axes[2, 0].set_title('Original Speech Signal')
                axes[2, 0].set_xlabel('Time (s)')
                axes[2, 0].set_ylabel('Amplitude')
                axes[2, 0].grid(True, alpha=0.3)
            
            # Спектрограммы
            D1 = librosa.stft(throat_audio[:sr1*3])
            D3 = librosa.stft(converted_audio[:sr3*3])
            
            im1 = axes[0, 1].imshow(librosa.amplitude_to_db(np.abs(D1)), 
                                   aspect='auto', origin='lower', cmap='viridis')
            axes[0, 1].set_title('Throat Microphone Spectrogram')
            axes[0, 1].set_xlabel('Time Frames')
            axes[0, 1].set_ylabel('Frequency Bins')
            plt.colorbar(im1, ax=axes[0, 1])
            
            im3 = axes[1, 1].imshow(librosa.amplitude_to_db(np.abs(D3)), 
                                   aspect='auto', origin='lower', cmap='viridis')
            axes[1, 1].set_title('Converted Speech Spectrogram')
            axes[1, 1].set_xlabel('Time Frames')
            axes[1, 1].set_ylabel('Frequency Bins')
            plt.colorbar(im3, ax=axes[1, 1])
            
            if speech_audio is not None:
                D2 = librosa.stft(speech_audio[:sr2*3])
                im2 = axes[2, 1].imshow(librosa.amplitude_to_db(np.abs(D2)), 
                                       aspect='auto', origin='lower', cmap='viridis')
                axes[2, 1].set_title('Original Speech Spectrogram')
                axes[2, 1].set_xlabel('Time Frames')
                axes[2, 1].set_ylabel('Frequency Bins')
                plt.colorbar(im2, ax=axes[2, 1])
            
            plt.tight_layout()
            
            # Сохранение визуализации
            comparison_path = output_dir / 'demo_comparison.png'
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Копирование аудиофайлов для демо
            import shutil
            shutil.copy2(throat_file, output_dir / 'input_throat.wav')
            if speech_file and Path(speech_file).exists():
                shutil.copy2(speech_file, output_dir / 'target_speech.wav')
            shutil.copy2(result['output_file'], output_dir / 'converted_speech.wav')
            
            print(f"✅ Демо-сравнение создано в {output_dir}")
            
            # Создание HTML демо
            self._create_html_demo(output_dir, result)
            
            return {
                'comparison_image': str(comparison_path),
                'input_audio': str(output_dir / 'input_throat.wav'),
                'target_audio': str(output_dir / 'target_speech.wav') if speech_file else None,
                'converted_audio': str(output_dir / 'converted_speech.wav'),
                'metrics': {k: v for k, v in result.items() if k in ['mse', 'mae', 'cosine_similarity', 'snr_db']}
            }
            
        except Exception as e:
            print(f"❌ Ошибка создания демо: {e}")
            return None
    
    def _create_html_demo(self, output_dir, result):
        """Создание HTML демо с аудиоплеерами"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VoiceX Demo Comparison</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 40px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 20px;
                }}
                .header h1 {{
                    color: #667eea;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                .audio-section {{ 
                    margin: 30px 0; 
                    padding: 25px; 
                    border: 2px solid #e0e0e0; 
                    border-radius: 15px;
                    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
                    transition: all 0.3s ease;
                }}
                .audio-section:hover {{
                    border-color: #667eea;
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
                }}
                .audio-section h3 {{
                    color: #495057;
                    font-size: 1.3em;
                    margin-bottom: 15px;
                }}
                audio {{
                    width: 100%;
                    height: 50px;
                    border-radius: 10px;
                }}
                .metrics {{ 
                    background: linear-gradient(145deg, #e3f2fd, #bbdefb); 
                    padding: 25px; 
                    margin: 30px 0; 
                    border-radius: 15px;
                    border-left: 5px solid #2196f3;
                }}
                .metrics h3 {{
                    color: #1976d2;
                    font-size: 1.4em;
                    margin-bottom: 20px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }}
                .metric-item {{
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #1976d2;
                }}
                img {{ 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    margin: 20px 0;
                }}
                .info-box {{
                    background: linear-gradient(145deg, #fff3e0, #ffe0b2);
                    border-left: 5px solid #ff9800;
                    padding: 25px;
                    margin: 30px 0;
                    border-radius: 15px;
                }}
                .emoji {{ font-size: 1.2em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><span class="emoji">🎙️</span> VoiceX Demo</h1>
                    <p>Демонстрация системы персонализации голоса</p>
                </div>
                
                <div class="audio-section">
                    <h3><span class="emoji">🎤</span> Вход: Горловой микрофон</h3>
                    <p>Сигнал, захваченный через вибрации кожи шеи</p>
                    <audio controls>
                        <source src="input_throat.wav" type="audio/wav">
                        Ваш браузер не поддерживает аудио элемент.
                    </audio>
                </div>
                
                <div class="audio-section">
                    <h3><span class="emoji">🔄</span> Выход: Персонализированная речь (VoiceX)</h3>
                    <p>Результат работы нейронной сети - восстановленный естественный голос</p>
                    <audio controls>
                        <source src="converted_speech.wav" type="audio/wav">
                        Ваш браузер не поддерживает аудио элемент.
                    </audio>
                </div>"""
        
        # Добавляем секцию с оригинальной речью, если она есть
        target_file = output_dir / 'target_speech.wav'
        if target_file.exists():
            html_content += """
                <div class="audio-section">
                    <h3><span class="emoji">🎯</span> Эталон: Оригинальная речь</h3>
                    <p>Целевая речь для сравнения качества</p>
                    <audio controls>
                        <source src="target_speech.wav" type="audio/wav">
                        Ваш браузер не поддерживает аудио элемент.
                    </audio>
                </div>"""
        
        # Добавляем метрики если они есть
        if 'mse' in result:
            html_content += f"""
                <div class="metrics">
                    <h3><span class="emoji">📊</span> Метрики качества</h3>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-value">{result['mse']:.6f}</div>
                            <div>MSE</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{result['mae']:.6f}</div>
                            <div>MAE</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{result['cosine_similarity']:.3f}</div>
                            <div>Схожесть</div>
                        </div>"""
            
            if 'snr_db' in result:
                html_content += f"""
                        <div class="metric-item">
                            <div class="metric-value">{result['snr_db']:.1f} dB</div>
                            <div>SNR</div>
                        </div>"""
            
            html_content += """
                    </div>
                </div>"""
        
        html_content += """
                <div>
                    <h3><span class="emoji">📈</span> Визуальное сравнение</h3>
                    <img src="demo_comparison.png" alt="Сравнение сигналов и спектрограмм">
                </div>
                
                <div class="info-box">
                    <h3><span class="emoji">🧠</span> Как это работает:</h3>
                    <ol>
                        <li><strong>Захват:</strong> Горловой микрофон улавливает вибрации через кожу</li>
                        <li><strong>Предобработка:</strong> Фильтрация шума и улучшение сигнала</li>
                        <li><strong>Извлечение признаков:</strong> Mel-спектрограммы, MFCC, pitch и другие</li>
                        <li><strong>ML персонализация:</strong> Нейронная сеть восстанавливает индивидуальный голос</li>
                        <li><strong>Синтез:</strong> Преобразование в натуральную речь</li>
                    </ol>
                    
                    <h4><span class="emoji">📋</span> Интерпретация метрик:</h4>
                    <ul>
                        <li><strong>MSE/MAE:</strong> Чем меньше, тем точнее восстановление</li>
                        <li><strong>Схожесть:</strong> Чем ближе к 1, тем больше похож на оригинал</li>
                        <li><strong>SNR:</strong> Чем выше, тем лучше качество сигнала</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: #666;">
                    <p><em>VoiceX - Восстановление голоса через персонализированное машинное обучение</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_path = output_dir / 'demo.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 HTML демо создано: {html_path}")

def main():
    """Основная функция тестирования"""
    parser = argparse.ArgumentParser(description='Тестирование модели VoiceX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к обученной модели (.h5 или .tflite)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Путь к файлу конфигурации модели')
    parser.add_argument('--scaler_path', type=str, required=True,
                       help='Путь к файлу скейлера')
    parser.add_argument('--data_path', type=str,
                       help='Путь к тестовым данным')
    parser.add_argument('--throat_file', type=str,
                       help='Файл горлового микрофона для единичного теста')
    parser.add_argument('--speech_file', type=str,
                       help='Файл обычной речи для единичного теста')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Количество образцов для тестирования')
    parser.add_argument('--demo', action='store_true',
                       help='Создать демо-сравнение')
    parser.add_argument('--test_dataset', action='store_true',
                       help='Тестировать на всем датасете')
    
    args = parser.parse_args()
    
    # Создание директории результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Запуск тестирования модели VoiceX")
    print("=" * 50)
    
    try:
        # Загрузка тестера
        tester = VoiceXModelTester(
            model_path=args.model_path,
            config_path=args.config_path,
            scaler_path=args.scaler_path
        )
        
        # Единичное тестирование
        if args.throat_file:
            print("🔄 Единичное тестирование...")
            result = tester.test_single_sample(args.throat_file, args.speech_file)
            
            if result:
                print("✅ Результат единичного теста:")
                print(f"   Входной файл: {result['input_file']}")
                print(f"   Выходной файл: {result.get('output_file', 'Не создан')}")
                print(f"   Успех конвертации: {result['conversion_success']}")
                print(f"   Размер входных признаков: {result.get('throat_features_shape', 'N/A')}")
                print(f"   Размер выходных признаков: {result.get('predicted_features_shape', 'N/A')}")
                
                if 'mse' in result:
                    print(f"   MSE: {result['mse']:.6f}")
                    print(f"   MAE: {result['mae']:.6f}")
                    print(f"   Косинусная схожесть: {result['cosine_similarity']:.3f}")
                    if 'snr_db' in result:
                        print(f"   SNR: {result['snr_db']:.2f} dB")
        
        # Создание демо
        if args.demo and args.throat_file:
            print("🔄 Создание демо-сравнения...")
            demo_result = tester.create_demo_comparison(
                throat_file=args.throat_file,
                speech_file=args.speech_file,
                output_dir=args.output_dir
            )
            
            if demo_result:
                print("✅ Демо-сравнение создано успешно!")
                print(f"   HTML демо: {args.output_dir}/demo.html")
                print(f"   Визуализация: {demo_result['comparison_image']}")
        
        # Тестирование датасета
        if args.test_dataset and args.data_path:
            print("🔄 Тестирование на датасете...")
            results = tester.test_dataset(
                data_path=args.data_path,
                output_dir=args.output_dir,
                num_samples=args.num_samples
            )
            
            if results:
                print("✅ Тестирование датасета завершено!")
                print(f"   Результаты сохранены в: {args.output_dir}")
                print(f"   HTML отчет: {args.output_dir}/test_report.html")
        
        print("=" * 50)
        print("🎉 Тестирование завершено успешно!")
        print(f"📁 Все результаты сохранены в: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())