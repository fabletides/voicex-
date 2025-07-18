
#include <Arduino.h>
#include <WiFi.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>
#include <esp_task_wdt.h>

// TensorFlow Lite для микроконтроллеров
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Пины и конфигурация
#define I2S_WS_PIN          25
#define I2S_SCK_PIN         26
#define I2S_SD_PIN          22
#define VIBRO_MOTOR_1_PIN   33
#define VIBRO_MOTOR_2_PIN   32
#define LED_STATUS_PIN      2
#define BUTTON_PIN          0

// Аудио параметры
#define SAMPLE_RATE         22050
#define CHANNELS            1
#define BITS_PER_SAMPLE     16
#define BUFFER_SIZE         1024
#define FEATURE_SIZE        512    // Размер входных признаков для ML
#define OUTPUT_SIZE         256    // Размер выходных признаков

// ML параметры
#define TENSOR_ARENA_SIZE   50000  // Размер памяти для TensorFlow Lite
#define MODEL_FILE_PATH     "/model.tflite"
#define CONFIG_FILE_PATH    "/config.json"

// Состояния системы
enum SystemState {
    STATE_INIT,
    STATE_READY,
    STATE_RECORDING,
    STATE_PROCESSING,
    STATE_PLAYING,
    STATE_ERROR
};

// Структуры данных
struct AudioBuffer {
    int16_t* data;
    size_t size;
    size_t write_index;
    size_t read_index;
    bool full;
};

struct MLConfig {
    float input_scale;
    int input_zero_point;
    float output_scale;
    int output_zero_point;
    int model_version;
    String user_id;
};

// Глобальные переменные
SystemState current_state = STATE_INIT;
AudioBuffer input_buffer;
AudioBuffer output_buffer;
MLConfig ml_config;

// TensorFlow Lite объекты
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

// Очереди для многозадачности
QueueHandle_t audio_input_queue;
QueueHandle_t feature_queue;
QueueHandle_t output_queue;

// Мьютексы для синхронизации
SemaphoreHandle_t ml_mutex;
SemaphoreHandle_t spiffs_mutex;

// Таски
TaskHandle_t audio_input_task_handle;
TaskHandle_t feature_extraction_task_handle;
TaskHandle_t ml_inference_task_handle;
TaskHandle_t audio_output_task_handle;

// Функции инициализации
bool init_spiffs() {
    Serial.println("🔧 Инициализация SPIFFS...");
    
    if (!SPIFFS.begin(true)) {
        Serial.println("❌ Ошибка монтирования SPIFFS");
        return false;
    }
    
    Serial.printf("📁 SPIFFS: используется %d из %d байт\n", 
                  SPIFFS.usedBytes(), SPIFFS.totalBytes());
    
    return true;
}

bool load_ml_config() {
    Serial.println("📋 Загрузка конфигурации ML...");
    
    xSemaphoreTake(spiffs_mutex, portMAX_DELAY);
    
    if (!SPIFFS.exists(CONFIG_FILE_PATH)) {
        Serial.println("⚠️ Файл конфигурации не найден, создается дефолтный");
        create_default_config();
    }
    
    File config_file = SPIFFS.open(CONFIG_FILE_PATH, "r");
    if (!config_file) {
        Serial.println("❌ Не удалось открыть файл конфигурации");
        xSemaphoreGive(spiffs_mutex);
        return false;
    }
    
    String config_content = config_file.readString();
    config_file.close();
    xSemaphoreGive(spiffs_mutex);
    
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, config_content);
    
    if (error) {
        Serial.printf("❌ Ошибка парсинга JSON: %s\n", error.c_str());
        return false;
    }
    
    ml_config.input_scale = doc["input_scale"] | 1.0f;
    ml_config.input_zero_point = doc["input_zero_point"] | 0;
    ml_config.output_scale = doc["output_scale"] | 1.0f;
    ml_config.output_zero_point = doc["output_zero_point"] | 0;
    ml_config.model_version = doc["model_version"] | 1;
    ml_config.user_id = doc["user_id"] | "default_user";
    
    Serial.println("✅ Конфигурация ML загружена");
    return true;
}

void create_default_config() {
    DynamicJsonDocument doc(1024);
    doc["input_scale"] = 1.0f;
    doc["input_zero_point"] = 0;
    doc["output_scale"] = 1.0f;
    doc["output_zero_point"] = 0;
    doc["model_version"] = 1;
    doc["user_id"] = "default_user";
    
    File config_file = SPIFFS.open(CONFIG_FILE_PATH, "w");
    if (config_file) {
        serializeJson(doc, config_file);
        config_file.close();
        Serial.println("✅ Создан дефолтный файл конфигурации");
    }
}

bool load_tflite_model() {
    Serial.println("🧠 Загрузка TensorFlow Lite модели...");
    
    xSemaphoreTake(spiffs_mutex, portMAX_DELAY);
    
    if (!SPIFFS.exists(MODEL_FILE_PATH)) {
        Serial.println("❌ Файл модели не найден в SPIFFS");
        Serial.println("💡 Загрузите model.tflite в SPIFFS через:");
        Serial.println("   - ESP32 Sketch Data Upload");
        Serial.println("   - Веб-интерфейс");
        Serial.println("   - OTA обновление");
        xSemaphoreGive(spiffs_mutex);
        return false;
    }
    
    File model_file = SPIFFS.open(MODEL_FILE_PATH, "r");
    if (!model_file) {
        Serial.println("❌ Не удалось открыть файл модели");
        xSemaphoreGive(spiffs_mutex);
        return false;
    }
    
    size_t model_size = model_file.size();
    Serial.printf("📊 Размер модели: %d байт\n", model_size);
    
    // Загрузка модели в память
    uint8_t* model_data = (uint8_t*)ps_malloc(model_size);
    if (!model_data) {
        Serial.println("❌ Не удалось выделить память для модели");
        model_file.close();
        xSemaphoreGive(spiffs_mutex);
        return false;
    }
    
    model_file.read(model_data, model_size);
    model_file.close();
    xSemaphoreGive(spiffs_mutex);
    
    // Инициализация TensorFlow Lite
    tflite_model = tflite::GetModel(model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("❌ Несовместимая версия схемы модели: %d (ожидается %d)\n",
                     tflite_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    // Создание резолвера операций
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    
    // Создание интерпретатора
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    interpreter = &static_interpreter;
    
    // Выделение тензоров
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("❌ Ошибка выделения тензоров");
        return false;
    }
    
    // Получение указателей на входные и выходные тензоры
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    Serial.printf("✅ Модель загружена успешно\n");
    Serial.printf("   Входной тензор: [%d, %d]\n", 
                  input_tensor->dims->data[0], input_tensor->dims->data[1]);
    Serial.printf("   Выходной тензор: [%d, %d]\n", 
                  output_tensor->dims->data[0], output_tensor->dims->data[1]);
    
    return true;
}

bool init_i2s() {
    Serial.println("🎤 Инициализация I2S...");
    
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN
    };
    
    esp_err_t result = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (result != ESP_OK) {
        Serial.printf("❌ Ошибка установки I2S драйвера: %s\n", esp_err_to_name(result));
        return false;
    }
    
    result = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (result != ESP_OK) {
        Serial.printf("❌ Ошибка настройки пинов I2S: %s\n", esp_err_to_name(result));
        return false;
    }
    
    Serial.println("✅ I2S инициализирован");
    return true;
}

bool init_audio_buffers() {
    Serial.println("🔊 Инициализация аудио буферов...");
    
    input_buffer.size = BUFFER_SIZE * 4; // 4 секунды буфера
    input_buffer.data = (int16_t*)ps_malloc(input_buffer.size * sizeof(int16_t));
    if (!input_buffer.data) {
        Serial.println("❌ Не удалось выделить память для входного буфера");
        return false;
    }
    
    output_buffer.size = BUFFER_SIZE * 4;
    output_buffer.data = (int16_t*)ps_malloc(output_buffer.size * sizeof(int16_t));
    if (!output_buffer.data) {
        Serial.println("❌ Не удалось выделить память для выходного буфера");
        return false;
    }
    
    // Инициализация буферов
    memset(input_buffer.data, 0, input_buffer.size * sizeof(int16_t));
    memset(output_buffer.data, 0, output_buffer.size * sizeof(int16_t));
    
    input_buffer.write_index = 0;
    input_buffer.read_index = 0;
    input_buffer.full = false;
    
    output_buffer.write_index = 0;
    output_buffer.read_index = 0;
    output_buffer.full = false;
    
    Serial.println("✅ Аудио буферы инициализированы");
    return true;
}

bool init_vibro_motors() {
    Serial.println("📳 Инициализация вибромоторов...");
    
    // Настройка PWM для вибромоторов
    ledcSetup(0, 1000, 8); // Канал 0, 1kHz, 8-bit разрешение
    ledcSetup(1, 1000, 8); // Канал 1, 1kHz, 8-bit разрешение
    
    ledcAttachPin(VIBRO_MOTOR_1_PIN, 0);
    ledcAttachPin(VIBRO_MOTOR_2_PIN, 1);
    
    // Тестовая вибрация
    ledcWrite(0, 128);
    ledcWrite(1, 128);
    delay(200);
    ledcWrite(0, 0);
    ledcWrite(1, 0);
    
    Serial.println("✅ Вибромоторы инициализированы");
    return true;
}

bool init_queues_and_mutexes() {
    Serial.println("🔄 Создание очередей и мьютексов...");
    
    audio_input_queue = xQueueCreate(10, sizeof(AudioBuffer*));
    feature_queue = xQueueCreate(5, sizeof(float*));
    output_queue = xQueueCreate(5, sizeof(float*));
    
    ml_mutex = xSemaphoreCreateMutex();
    spiffs_mutex = xSemaphoreCreateMutex();
    
    if (!audio_input_queue || !feature_queue || !output_queue || 
        !ml_mutex || !spiffs_mutex) {
        Serial.println("❌ Ошибка создания очередей/мьютексов");
        return false;
    }
    
    Serial.println("✅ Очереди и мьютексы созданы");
    return true;
}

// Извлечение признаков из аудио сигнала
void extract_features(int16_t* audio_data, size_t length, float* features) {
    // Упрощенное извлечение признаков для ESP32
    // В реальной реализации здесь будет FFT, MFCC и другие алгоритмы
    
    // 1. Нормализация
    float max_val = 0;
    for (size_t i = 0; i < length; i++) {
        max_val = max(max_val, abs(audio_data[i]));
    }
    
    if (max_val == 0) max_val = 1; // Избегаем деления на ноль
    
    // 2. Базовые статистические признаки
    float mean = 0, variance = 0;
    for (size_t i = 0; i < length; i++) {
        float normalized = audio_data[i] / max_val;
        mean += normalized;
        features[i % FEATURE_SIZE] = normalized; // Циклическое заполнение
    }
    mean /= length;
    
    for (size_t i = 0; i < length; i++) {
        float normalized = audio_data[i] / max_val;
        variance += (normalized - mean) * (normalized - mean);
    }
    variance /= length;
    
    // 3. Энергетические признаки
    float energy = 0;
    for (size_t i = 0; i < length; i++) {
        float normalized = audio_data[i] / max_val;
        energy += normalized * normalized;
    }
    
    // Добавляем статистические признаки в конец массива
    if (FEATURE_SIZE > 3) {
        features[FEATURE_SIZE - 3] = mean;
        features[FEATURE_SIZE - 2] = sqrt(variance);
        features[FEATURE_SIZE - 1] = energy / length;
    }
}

// Предобработка сигнала горлового микрофона
void preprocess_throat_signal(int16_t* input, int16_t* output, size_t length) {
    // Простой полосовой фильтр для голосовых частот
    // В реальной реализации использовать IIR/FIR фильтры
    
    float alpha = 0.1; // Коэффициент сглаживания
    int16_t prev_sample = 0;
    
    for (size_t i = 0; i < length; i++) {
        // Высокочастотный фильтр (убираем DC компоненту)
        int16_t hp_output = input[i] - prev_sample;
        prev_sample = input[i];
        
        // Сглаживание (простой НЧ фильтр)
        static int16_t lp_prev = 0;
        lp_prev = alpha * hp_output + (1 - alpha) * lp_prev;
        
        output[i] = lp_prev;
    }
}

// ML Inference
bool run_ml_inference(float* input_features, float* output_features) {
    if (!interpreter || !input_tensor || !output_tensor) {
        Serial.println("❌ ML модель не загружена");
        return false;
    }
    
    xSemaphoreTake(ml_mutex, portMAX_DELAY);
    
    // Квантизация входных данных если нужно
    if (input_tensor->type == kTfLiteInt8) {
        for (int i = 0; i < FEATURE_SIZE; i++) {
            input_tensor->data.int8[i] = 
                (input_features[i] / ml_config.input_scale) + ml_config.input_zero_point;
        }
    } else {
        // Float32 модель
        for (int i = 0; i < FEATURE_SIZE; i++) {
            input_tensor->data.f[i] = input_features[i];
        }
    }
    
    // Выполнение inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("❌ Ошибка выполнения ML inference");
        xSemaphoreGive(ml_mutex);
        return false;
    }
    
    // Деквантизация выходных данных если нужно
    if (output_tensor->type == kTfLiteInt8) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_features[i] = 
                (output_tensor->data.int8[i] - ml_config.output_zero_point) * ml_config.output_scale;
        }
    } else {
        // Float32 модель
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_features[i] = output_tensor->data.f[i];
        }
    }
    
    xSemaphoreGive(ml_mutex);
    return true;
}

// Преобразование признаков в аудио сигнал для вибромоторов
void features_to_vibration(float* features, uint8_t* vibro1_signal, uint8_t* vibro2_signal, size_t length) {
    for (size_t i = 0; i < length; i++) {
        // Разделяем признаки между двумя моторами для создания стерео эффекта
        float feature1 = features[i % OUTPUT_SIZE];
        float feature2 = features[(i + OUTPUT_SIZE/2) % OUTPUT_SIZE];
        
        // Нормализация в диапазон 0-255 для PWM
        vibro1_signal[i] = constrain(abs(feature1) * 255, 0, 255);
        vibro2_signal[i] = constrain(abs(feature2) * 255, 0, 255);
    }
}

// Задача захвата аудио
void audio_input_task(void* parameter) {
    Serial.println("🎤 Запуск задачи захвата аудио");
    
    int16_t* temp_buffer = (int16_t*)malloc(BUFFER_SIZE * sizeof(int16_t));
    if (!temp_buffer) {
        Serial.println("❌ Не удалось выделить память для временного буфера");
        vTaskDelete(NULL);
        return;
    }
    
    while (true) {
        size_t bytes_read = 0;
        esp_err_t result = i2s_read(I2S_NUM_0, temp_buffer, 
                                   BUFFER_SIZE * sizeof(int16_t), 
                                   &bytes_read, portMAX_DELAY);
        
        if (result == ESP_OK && bytes_read > 0) {
            size_t samples_read = bytes_read / sizeof(int16_t);
            
            // Предобработка сигнала горлового микрофона
            preprocess_throat_signal(temp_buffer, temp_buffer, samples_read);
            
            // Копирование в кольцевой буфер
            for (size_t i = 0; i < samples_read; i++) {
                input_buffer.data[input_buffer.write_index] = temp_buffer[i];
                input_buffer.write_index = (input_buffer.write_index + 1) % input_buffer.size;
                
                if (input_buffer.write_index == input_buffer.read_index) {
                    input_buffer.full = true;
                    input_buffer.read_index = (input_buffer.read_index + 1) % input_buffer.size;
                }
            }
            
            // Отправка сигнала о новых данных
            AudioBuffer* buffer_ptr = &input_buffer;
            xQueueSend(audio_input_queue, &buffer_ptr, 0);
        }
        
        esp_task_wdt_reset(); // Сброс watchdog
        vTaskDelay(1); // Небольшая задержка
    }
    
    free(temp_buffer);
    vTaskDelete(NULL);
}

// Задача извлечения признаков
void feature_extraction_task(void* parameter) {
    Serial.println("🔍 Запуск задачи извлечения признаков");
    
    float* features = (float*)malloc(FEATURE_SIZE * sizeof(float));
    if (!features) {
        Serial.println("❌ Не удалось выделить память для признаков");
        vTaskDelete(NULL);
        return;
    }
    
    AudioBuffer* received_buffer;
    
    while (true) {
        if (xQueueReceive(audio_input_queue, &received_buffer, portMAX_DELAY)) {
            // Извлекаем признаки из последних данных в буфере
            size_t start_index = received_buffer->write_index > BUFFER_SIZE ? 
                                received_buffer->write_index - BUFFER_SIZE : 0;
            
            extract_features(received_buffer->data + start_index, 
                           min(BUFFER_SIZE, received_buffer->write_index), 
                           features);
            
            // Отправляем признаки в очередь для ML
            xQueueSend(feature_queue, &features, 0);
        }
        
        esp_task_wdt_reset();
        vTaskDelay(1);
    }
    
    free(features);
    vTaskDelete(NULL);
}

// Задача ML inference
void ml_inference_task(void* parameter) {
    Serial.println("🧠 Запуск задачи ML inference");
    
    float* input_features;
    float* output_features = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    if (!output_features) {
        Serial.println("❌ Не удалось выделить память для выходных признаков");
        vTaskDelete(NULL);
        return;
    }
    
    while (true) {
        if (xQueueReceive(feature_queue, &input_features, portMAX_DELAY)) {
            if (run_ml_inference(input_features, output_features)) {
                // Отправляем результат в очередь воспроизведения
                xQueueSend(output_queue, &output_features, 0);
            } else {
                Serial.println("⚠️ Ошибка ML inference, пропускаем фрейм");
                
                // Попытка восстановления модели
                if (!interpreter) {
                    Serial.println("🔄 Попытка перезагрузки модели...");
                    if (load_tflite_model()) {
                        Serial.println("✅ Модель восстановлена");
                    }
                }
            }
        }
        
        esp_task_wdt_reset();
        vTaskDelay(1);
    }
    
    free(output_features);
    vTaskDelete(NULL);
}

// Задача воспроизведения через вибромоторы
void audio_output_task(void* parameter) {
    Serial.println("📳 Запуск задачи воспроизведения");
    
    uint8_t* vibro1_signal = (uint8_t*)malloc(OUTPUT_SIZE);
    uint8_t* vibro2_signal = (uint8_t*)malloc(OUTPUT_SIZE);
    
    if (!vibro1_signal || !vibro2_signal) {
        Serial.println("❌ Не удалось выделить память для сигналов вибромоторов");
        vTaskDelete(NULL);
        return;
    }
    
    float* received_features;
    
    while (true) {
        if (xQueueReceive(output_queue, &received_features, portMAX_DELAY)) {
            // Преобразуем признаки в сигналы для вибромоторов
            features_to_vibration(received_features, vibro1_signal, vibro2_signal, OUTPUT_SIZE);
            
            // Воспроизводим через вибромоторы
            for (size_t i = 0; i < OUTPUT_SIZE; i++) {
                ledcWrite(0, vibro1_signal[i]);
                ledcWrite(1, vibro2_signal[i]);
                
                // Задержка для создания временной развертки
                delayMicroseconds(100); // 10kHz частота обновления
            }
        }
        
        esp_task_wdt_reset();
        vTaskDelay(1);
    }
    
    free(vibro1_signal);
    free(vibro2_signal);
    vTaskDelete(NULL);
}

// Функция обработки ошибок и восстановления
void handle_system_error(const char* error_msg) {
    Serial.printf("❌ Системная ошибка: %s\n", error_msg);
    
    current_state = STATE_ERROR;
    digitalWrite(LED_STATUS_PIN, HIGH); // Индикация ошибки
    
    // Останавливаем вибромоторы
    ledcWrite(0, 0);
    ledcWrite(1, 0);
    
    // Попытка восстановления через 5 секунд
    delay(5000);
    Serial.println("🔄 Попытка восстановления системы...");
    
    // Перезапуск критически важных компонентов
    if (!interpreter) {
        load_tflite_model();
    }
    
    current_state = STATE_READY;
    digitalWrite(LED_STATUS_PIN, LOW);
    Serial.println("✅ Система восстановлена");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("🚀 Запуск VoiceX - Система восстановления голоса");
    Serial.println("===============================================");
    
    // Инициализация пинов
    pinMode(LED_STATUS_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    // Индикация запуска
    digitalWrite(LED_STATUS_PIN, HIGH);
    
    current_state = STATE_INIT;
    
    // Пошаговая инициализация всех компонентов
    bool init_success = true;
    
    // 1. SPIFFS
    if (!init_spiffs()) {
        handle_system_error("Инициализация SPIFFS");
        init_success = false;
    }
    
    // 2. Конфигурация ML
    if (init_success && !load_ml_config()) {
        handle_system_error("Загрузка конфигурации ML");
        init_success = false;
    }
    
    // 3. TensorFlow Lite модель
    if (init_success && !load_tflite_model()) {
        handle_system_error("Загрузка TensorFlow Lite модели");
        init_success = false;
    }
    
    // 4. I2S аудио
    if (init_success && !init_i2s()) {
        handle_system_error("Инициализация I2S");
        init_success = false;
    }
    
    // 5. Аудио буферы
    if (init_success && !init_audio_buffers()) {
        handle_system_error("Инициализация аудио буферов");
        init_success = false;
    }
    
    // 6. Вибромоторы
    if (init_success && !init_vibro_motors()) {
        handle_system_error("Инициализация вибромоторов");
        init_success = false;
    }
    
    // 7. Очереди и мьютексы
    if (init_success && !init_queues_and_mutexes()) {
        handle_system_error("Создание очередей и мьютексов");
        init_success = false;
    }
    
    if (init_success) {
        // Создание задач
        Serial.println("🔄 Создание задач...");
        
        xTaskCreatePinnedToCore(
            audio_input_task,
            "AudioInput",
            4096,
            NULL,
            2, // Высокий приоритет для аудио
            &audio_input_task_handle,
            0  // Ядро 0
        );
        
        xTaskCreatePinnedToCore(
            feature_extraction_task,
            "FeatureExtraction", 
            4096,
            NULL,
            1,
            &feature_extraction_task_handle,
            0  // Ядро 0
        );
        
        xTaskCreatePinnedToCore(
            ml_inference_task,
            "MLInference",
            8192, // Больше стека для ML
            NULL,
            1,
            &ml_inference_task_handle,
            1  // Ядро 1 для ML
        );
        
        xTaskCreatePinnedToCore(
            audio_output_task,
            "AudioOutput",
            4096,
            NULL,
            2, // Высокий приоритет для вывода
            &audio_output_task_handle,
            1  // Ядро 1
        );
        
        current_state = STATE_READY;
        digitalWrite(LED_STATUS_PIN, LOW);
        
        Serial.println("✅ VoiceX система готова к работе!");
        Serial.println("📱 Нажмите кнопку для начала записи");
    } else {
        current_state = STATE_ERROR;
        Serial.println("❌ Инициализация не удалась. Система в режиме ошибки.");
    }
    
    // Настройка Watchdog
    esp_task_wdt_init(30, true); // 30 секунд таймаут
    esp_task_wdt_add(NULL); // Добавляем основную задачу
}

void loop() {
    static unsigned long last_status_update = 0;
    static unsigned long last_button_check = 0;
    static bool button_pressed = false;
    static unsigned long button_press_time = 0;
    
    unsigned long current_time = millis();
    
    // Проверка кнопки каждые 50мс
    if (current_time - last_button_check > 50) {
        bool current_button_state = digitalRead(BUTTON_PIN) == LOW;
        
        if (current_button_state && !button_pressed) {
            // Начало нажатия
            button_pressed = true;
            button_press_time = current_time;
            
            if (current_state == STATE_READY) {
                current_state = STATE_RECORDING;
                Serial.println("🎤 Начало записи...");
                digitalWrite(LED_STATUS_PIN, HIGH);
            }
        } else if (!current_button_state && button_pressed) {
            // Отпускание кнопки
            button_pressed = false;
            unsigned long press_duration = current_time - button_press_time;
            
            if (current_state == STATE_RECORDING) {
                if (press_duration > 100) { // Минимум 100мс
                    current_state = STATE_PROCESSING;
                    Serial.printf("🔄 Обработка записи (%lu мс)...\n", press_duration);
                } else {
                    current_state = STATE_READY;
                    Serial.println("⚠️ Слишком короткая запись, игнорируем");
                }
                digitalWrite(LED_STATUS_PIN, LOW);
            }
            
            // Длинное нажатие (>3 сек) для особых функций
            if (press_duration > 3000) {
                Serial.println("🔧 Длинное нажатие - перезагрузка системы");
                ESP.restart();
            }
        }
        
        last_button_check = current_time;
    }
    
    // Обновление статуса каждые 5 секунд
    if (current_time - last_status_update > 5000) {
        print_system_status();
        last_status_update = current_time;
    }
    
    // Управление индикацией состояния
    switch (current_state) {
        case STATE_INIT:
            // Быстрое мигание при инициализации
            digitalWrite(LED_STATUS_PIN, (millis() / 200) % 2);
            break;
            
        case STATE_READY:
            // Медленное мигание в режиме готовности
            digitalWrite(LED_STATUS_PIN, (millis() / 1000) % 2);
            break;
            
        case STATE_RECORDING:
            // Постоянное свечение при записи
            digitalWrite(LED_STATUS_PIN, HIGH);
            break;
            
        case STATE_PROCESSING:
            // Быстрое мигание при обработке
            digitalWrite(LED_STATUS_PIN, (millis() / 100) % 2);
            // Автоматический переход в READY после обработки
            if (current_time - button_press_time > 2000) {
                current_state = STATE_READY;
            }
            break;
            
        case STATE_PLAYING:
            // Двойное мигание при воспроизведении
            int pattern = (millis() / 150) % 4;
            digitalWrite(LED_STATUS_PIN, pattern < 2 ? HIGH : LOW);
            break;
            
        case STATE_ERROR:
            // SOS сигнал при ошибке
            int sos_pattern = (millis() / 200) % 8;
            bool sos_state = (sos_pattern < 3) || (sos_pattern == 4) || (sos_pattern == 5);
            digitalWrite(LED_STATUS_PIN, sos_state ? HIGH : LOW);
            break;
    }
    
    // Мониторинг использования памяти
    static unsigned long last_memory_check = 0;
    if (current_time - last_memory_check > 10000) { // Каждые 10 сек
        size_t free_heap = esp_get_free_heap_size();
        size_t min_free_heap = esp_get_minimum_free_heap_size();
        
        if (free_heap < 10000) { // Критично мало памяти
            Serial.printf("⚠️ Критично мало памяти: %d байт\n", free_heap);
            handle_system_error("Недостаток памяти");
        }
        
        last_memory_check = current_time;
    }
    
    // Проверка состояния задач
    static unsigned long last_task_check = 0;
    if (current_time - last_task_check > 15000) { // Каждые 15 сек
        check_task_health();
        last_task_check = current_time;
    }
    
    // Сброс watchdog
    esp_task_wdt_reset();
    
    // Небольшая задержка для снижения нагрузки на CPU
    delay(10);
}

void print_system_status() {
    size_t free_heap = esp_get_free_heap_size();
    size_t min_free_heap = esp_get_minimum_free_heap_size();
    
    Serial.println("📊 Статус системы VoiceX:");
    Serial.printf("   Состояние: %s\n", get_state_name(current_state));
    Serial.printf("   Свободная память: %d байт (мин: %d)\n", free_heap, min_free_heap);
    Serial.printf("   Время работы: %lu сек\n", millis() / 1000);
    
    // Статус очередей
    if (audio_input_queue) {
        Serial.printf("   Очередь аудио входа: %d/%d\n", 
                     uxQueueMessagesWaiting(audio_input_queue), 10);
    }
    if (feature_queue) {
        Serial.printf("   Очередь признаков: %d/%d\n", 
                     uxQueueMessagesWaiting(feature_queue), 5);
    }
    if (output_queue) {
        Serial.printf("   Очередь вывода: %d/%d\n", 
                     uxQueueMessagesWaiting(output_queue), 5);
    }
    
    // Состояние ML модели
    Serial.printf("   ML модель: %s\n", interpreter ? "Загружена" : "Не загружена");
    Serial.printf("   Пользователь: %s\n", ml_config.user_id.c_str());
    
    Serial.println("---");
}

const char* get_state_name(SystemState state) {
    switch (state) {
        case STATE_INIT: return "Инициализация";
        case STATE_READY: return "Готов";
        case STATE_RECORDING: return "Запись";
        case STATE_PROCESSING: return "Обработка";
        case STATE_PLAYING: return "Воспроизведение";
        case STATE_ERROR: return "Ошибка";
        default: return "Неизвестно";
    }
}

void check_task_health() {
    // Проверка состояния задач
    TaskHandle_t tasks[] = {
        audio_input_task_handle,
        feature_extraction_task_handle,
        ml_inference_task_handle,
        audio_output_task_handle
    };
    
    const char* task_names[] = {
        "AudioInput",
        "FeatureExtraction", 
        "MLInference",
        "AudioOutput"
    };
    
    bool all_tasks_healthy = true;
    
    for (int i = 0; i < 4; i++) {
        if (tasks[i] != NULL) {
            eTaskState task_state = eTaskGetState(tasks[i]);
            if (task_state == eDeleted || task_state == eInvalid) {
                Serial.printf("❌ Задача %s не работает (состояние: %d)\n", task_names[i], task_state);
                all_tasks_healthy = false;
                
                // Попытка перезапуска задачи
                restart_task(i);
            }
        } else {
            Serial.printf("❌ Задача %s не существует\n", task_names[i]);
            all_tasks_healthy = false;
        }
    }
    
    if (!all_tasks_healthy) {
        Serial.println("⚠️ Обнаружены проблемы с задачами");
    }
}

void restart_task(int task_index) {
    Serial.printf("🔄 Перезапуск задачи %d...\n", task_index);
    
    // Удаляем старую задачу если она еще существует
    TaskHandle_t* task_handles[] = {
        &audio_input_task_handle,
        &feature_extraction_task_handle,
        &ml_inference_task_handle,
        &audio_output_task_handle
    };
    
    if (*task_handles[task_index] != NULL) {
        vTaskDelete(*task_handles[task_index]);
        *task_handles[task_index] = NULL;
    }
    
    // Создаем задачу заново
    switch (task_index) {
        case 0:
            xTaskCreatePinnedToCore(audio_input_task, "AudioInput", 4096, NULL, 2, 
                                   &audio_input_task_handle, 0);
            break;
        case 1:
            xTaskCreatePinnedToCore(feature_extraction_task, "FeatureExtraction", 4096, NULL, 1,
                                   &feature_extraction_task_handle, 0);
            break;
        case 2:
            xTaskCreatePinnedToCore(ml_inference_task, "MLInference", 8192, NULL, 1,
                                   &ml_inference_task_handle, 1);
            break;
        case 3:
            xTaskCreatePinnedToCore(audio_output_task, "AudioOutput", 4096, NULL, 2,
                                   &audio_output_task_handle, 1);
            break;
    }
    
    Serial.printf("✅ Задача %d перезапущена\n", task_index);
}

// Дополнительные утилиты для отладки и диагностики

void dump_system_info() {
    Serial.println("🔍 Полная информация о системе:");
    Serial.println("================================");
    
    // Информация о чипе
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    Serial.printf("Чип: %s rev %d\n", CONFIG_IDF_TARGET, chip_info.revision);
    Serial.printf("Ядра: %d\n", chip_info.cores);
    Serial.printf("WiFi: %s, Bluetooth: %s\n", 
                  (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "Да" : "Нет",
                  (chip_info.features & CHIP_FEATURE_BT) ? "Да" : "Нет");
    
    // Информация о памяти
    Serial.printf("Общая память: %d байт\n", ESP.getHeapSize());
    Serial.printf("Свободная память: %d байт\n", ESP.getFreeHeap());
    Serial.printf("Минимальная свободная: %d байт\n", ESP.getMinFreeHeap());
    Serial.printf("Максимальный блок: %d байт\n", ESP.getMaxAllocHeap());
    
    // Информация о Flash
    Serial.printf("Flash размер: %d байт\n", ESP.getFlashChipSize());
    Serial.printf("Flash скорость: %d Hz\n", ESP.getFlashChipSpeed());
    
    // SPIFFS информация
    Serial.printf("SPIFFS общий: %d байт\n", SPIFFS.totalBytes());
    Serial.printf("SPIFFS используется: %d байт\n", SPIFFS.usedBytes());
    
    Serial.println("================================");
}

void save_debug_log() {
    // Сохранение лога отладки в SPIFFS для анализа
    xSemaphoreTake(spiffs_mutex, portMAX_DELAY);
    
    File debug_file = SPIFFS.open("/debug.log", "a");
    if (debug_file) {
        debug_file.printf("Время: %lu, Состояние: %s, Память: %d\n",
                         millis(), get_state_name(current_state), ESP.getFreeHeap());
        debug_file.close();
    }
    
    xSemaphoreGive(spiffs_mutex);
}