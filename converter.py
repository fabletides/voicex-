#!/usr/bin/env python3
"""
VoiceX Model to Header Converter
Конвертирует .tflite модель в C++ заголовочный файл для встраивания в ESP32
"""

import argparse
import os
from pathlib import Path
import json
from datetime import datetime

def convert_tflite_to_header(tflite_path, output_path, var_name="voicex_model"):
    """Конвертирует .tflite файл в C++ заголовок"""
    
    print(f"🔄 Конвертирование {tflite_path} в {output_path}")
    
    # Чтение файла модели
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    print(f"📊 Размер модели: {model_size} байт ({model_size/1024:.2f} KB)")
    
    # Проверка размера (ESP32 ограничения)
    if model_size > 500000:  # 500KB
        print("⚠️ Предупреждение: модель больше 500KB, может не поместиться в ESP32")
    
    # Генерация заголовочного файла
    header_content = generate_header_content(model_data, var_name, tflite_path)
    
    # Сохранение
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"✅ Заголовочный файл создан: {output_path}")
    print(f"📝 Размер данных: {model_size} байт")
    
    return output_path

def generate_header_content(model_data, var_name, source_path):
    """Генерирует содержимое C++ заголовочного файла"""
    
    model_size = len(model_data)
    guard_name = f"{var_name.upper()}_H"
    
    # Извлечение метаданных из пути файла
    source_file = Path(source_path)
    creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f'''#ifndef {guard_name}
#define {guard_name}

/*
 * VoiceX TensorFlow Lite Model Data
 * Автоматически сгенерировано из: {source_file.name}
 * Размер: {model_size} байт ({model_size/1024:.2f} KB)
 * Создано: {creation_date}
 * 
 * Использование:
 *   #include "{source_file.stem}_data.h"
 *   const tflite::Model* model = tflite::GetModel({var_name}_data);
 */

#include <stdint.h>

// Размер модели
const unsigned int {var_name}_len = {model_size};

// Данные модели
alignas(8) const unsigned char {var_name}_data[] = {{
'''
    
    # Добавление байтов модели
    for i, byte in enumerate(model_data):
        if i % 12 == 0:
            header += '\n  '
        header += f'0x{byte:02x}, '
    
    # Удаление последней запятой и добавление закрывающей скобки
    header = header.rstrip(', ') + '\n};\n\n'
    
    # Добавление вспомогательных функций
    helper_functions = f'''// Вспомогательные функции
const unsigned char* get_{var_name}_data() {{
    return {var_name}_data;
}}

unsigned int get_{var_name}_size() {{
    return {var_name}_len;
}}

// Проверка валидности модели TensorFlow Lite
bool validate_{var_name}() {{
    if ({var_name}_len < 8) return false;
    
    // Проверка магических байтов TFLite
    const char* magic = "TFL3";
    for (int i = 0; i < 4; i++) {{
        if ({var_name}_data[i + 4] != magic[i]) {{
            return false;
        }}
    }}
    return true;
}}

// Метаданные модели
struct {var_name.title()}Metadata {{
    const char* source_file = "{source_file.name}";
    const char* created_date = "{creation_date}";
    unsigned int size_bytes = {model_size};
    float size_kb = {model_size/1024:.2f}f;
    const char* description = "VoiceX TensorFlow Lite model for voice personalization";
    const char* version = "1.0.0";
}};

extern const {var_name.title()}Metadata {var_name}_metadata;
const {var_name.title()}Metadata {var_name}_metadata = {{}};

#endif // {guard_name}
'''
    
    return header + helper_functions

def create_demo_model():
    """Создает демо TensorFlow Lite модель для тестирования"""
    
    try:
        import tensorflow as tf
        import numpy as np
        
        print("🔧 Создание демо TensorFlow Lite модели...")
        
        # Простая модель autoencoder
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(512,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Создание фиктивных данных для representative dataset
        def representative_dataset():
            for _ in range(100):
                yield [np.random.random((1, 512)).astype(np.float32)]
        
        # Конвертация в TFLite с квантизацией
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Сохранение
        demo_path = "demo_voicex_model.tflite"
        with open(demo_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Демо модель создана: {demo_path} ({len(tflite_model)} байт)")
        return demo_path
        
    except ImportError:
        print("❌ TensorFlow не установлен, создание демо модели невозможно")
        return None

def verify_header_file(header_path):
    """Проверяет корректность созданного заголовочного файла"""
    
    print(f"🔍 Проверка заголовочного файла: {header_path}")
    
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        
        # Базовые проверки
        checks = [
            ('#ifndef' in content, "Include guard"),
            ('#define' in content, "Define guard"),
            ('const unsigned char' in content, "Data array"),
            ('const unsigned int' in content, "Size variable"),
            ('0x' in content, "Hex data"),
            ('#endif' in content, "End guard")
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description}")
                all_passed = False
        
        if all_passed:
            print("✅ Заголовочный файл прошел все проверки")
        else:
            print("❌ Заголовочный файл содержит ошибки")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Ошибка при проверке файла: {e}")
        return False

def create_build_instructions(output_path, header_file):
    """Создает инструкции по сборке для ESP32"""
    
    instructions = f"""
# Инструкции по использованию модели в ESP32

## 1. Скопируйте заголовочный файл
```bash
cp {header_file} esp32_firmware/src/
```

## 2. Включите в main.cpp
```cpp
#include "{Path(header_file).name}"

// В функции load_tflite_model() замените:
// tflite_model = tflite::GetModel(model_data);
// на:
tflite_model = tflite::GetModel(voicex_model_data);

// Проверка модели:
if (!validate_voicex_model()) {{
    Serial.println("❌ Встроенная модель повреждена");
    return false;
}}
```

## 3. Обновите конфигурацию
```cpp
// Размеры модели
#define MODEL_SIZE {Path(header_file).stat().st_size if Path(header_file).exists() else 'XXX'}
#define TENSOR_ARENA_SIZE 60000  // Увеличьте если нужно
```

## 4. Компиляция
```bash
cd esp32_firmware
pio run
pio run --target upload
```

## 5. Проверка в Serial Monitor
Должны увидеть:
```
✅ Модель загружена: встроенная
📊 Тип модели: tflite
🎯 Входной тензор: [1, 512]
🎯 Выходной тензор: [1, 256]
```

## Замечания
- Встроенная модель увеличивает размер прошивки
- Альтернатива: загрузка через SPIFFS
- Для обновления модели нужна перепрошивка
"""
    
    instructions_path = output_path.replace('.h', '_instructions.md')
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📋 Инструкции созданы: {instructions_path}")
    return instructions_path

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Конвертер TFLite модели в C++ заголовок')
    parser.add_argument('--model', type=str, 
                       help='Путь к .tflite модели')
    parser.add_argument('--output', type=str, default='voicex_model_data.h',
                       help='Выходной .h файл')
    parser.add_argument('--var_name', type=str, default='voicex_model',
                       help='Имя переменной в C++')
    parser.add_argument('--create_demo', action='store_true',
                       help='Создать демо модель если TensorFlow доступен')
    parser.add_argument('--verify', action='store_true',
                       help='Проверить созданный заголовочный файл')
    
    args = parser.parse_args()
    
    print("🔧 VoiceX Model to Header Converter")
    print("=" * 40)
    
    # Создание демо модели если запрошено
    if args.create_demo:
        demo_model = create_demo_model()
        if demo_model and not args.model:
            args.model = demo_model
    
    # Проверка входного файла
    if not args.model:
        print("❌ Не указан файл модели")
        print("💡 Используйте --model путь/к/модели.tflite")
        print("💡 Или --create_demo для создания тестовой модели")
        return 1
    
    if not os.path.exists(args.model):
        print(f"❌ Файл модели не найден: {args.model}")
        return 1
    
    # Конвертация
    try:
        header_file = convert_tflite_to_header(
            args.model, 
            args.output, 
            args.var_name
        )
        
        # Проверка если запрошено
        if args.verify:
            verify_header_file(header_file)
        
        # Создание инструкций
        create_build_instructions(args.output, header_file)
        
        print("=" * 40)
        print("🎉 Конвертация завершена!")
        print(f"📁 Заголовочный файл: {header_file}")
        print(f"📋 Инструкции: {args.output.replace('.h', '_instructions.md')}")
        print("")
        print("📝 Следующие шаги:")
        print(f"   1. cp {header_file} esp32_firmware/src/")
        print(f"   2. Включите #{args.var_name}_data.h в main.cpp")
        print("   3. Замените загрузку из SPIFFS на встроенные данные")
        print("   4. Компилируйте и загрузите прошивку")
        
        return 0
        
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())