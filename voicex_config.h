// ===== VOICEX_CONFIG.H - КОНФИГУРАЦИЯ =====
#ifndef VOICEX_CONFIG_H
#define VOICEX_CONFIG_H

#include <Arduino.h>
#include <WiFi.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <SPIFFS.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_INA219.h>
#include <ArduinoJson.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <ArduinoOTA.h>
#include <esp_task_wdt.h>
#include <driver/dac.h>
#include <HTTPClient.h>

// ===== HARDWARE CONFIGURATION =====
// GPIO Pin Definitions
#define VIBRO_MOTOR_1_PIN       25
#define VIBRO_MOTOR_2_PIN       26
#define SOS_BUTTON_PIN          13
#define BATTERY_MONITOR_SDA     21
#define BATTERY_MONITOR_SCL     22
#define OLED_SDA                21
#define OLED_SCL                22
#define I2S_WS                  15
#define I2S_SD                  2
#define I2S_SCK                 14
#define I2S_SD_OUT              4
#define PROXIMITY_SENSOR_PIN    27
#define STATUS_LED_PIN          5

// ===== AUDIO CONFIGURATION =====
#define I2S_SAMPLE_RATE         22050    // Higher quality for voice
#define VOICE_SAMPLE_RATE       22050
#define I2S_CHANNELS            1
#define I2S_BITS_PER_SAMPLE     16
#define I2S_BUFFER_SIZE         1024
#define I2S_BUFFER_COUNT        8
#define VOICE_FRAME_SIZE        1024
#define VOICE_HOP_SIZE          512
#define MEL_BINS                80
#define N_FFT                   2048

// ===== BLE CONFIGURATION =====
#define BLE_DEVICE_NAME         "VoiceX_Enhanced"
#define BLE_SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define BLE_CHAR_AUDIO_UUID     "12345678-1234-1234-1234-123456789abd"
#define BLE_CHAR_STATUS_UUID    "12345678-1234-1234-1234-123456789abe"
#define BLE_CHAR_CONTROL_UUID   "12345678-1234-1234-1234-123456789abf"

// ===== ML CONFIGURATION =====
#define ML_VOICE_CONVERSION_MODEL_SIZE   200000
#define ML_PERSONALIZATION_MODEL_SIZE    100000
#define ML_TENSOR_ARENA_SIZE            150000
#define ML_INPUT_SIZE                   1024
#define ML_OUTPUT_SIZE                  1024
#define VOICE_FEATURES_SIZE             128
#define PERSONALIZATION_FEATURES_SIZE   64

// ===== PERSONALIZATION CONFIGURATION =====
#define MAX_VOICE_PROFILES             5
#define VOICE_CALIBRATION_SAMPLES      100
#define ADAPTATION_LEARNING_RATE       0.01f
#define VOICE_SIMILARITY_THRESHOLD     0.85f

// ===== DEVICE ENUMS =====
enum DeviceMode {
    AUTO_MODE,
    MANUAL_MODE,
    LOUD_MODE,        // Добавлен недостающий режим
    SOS_MODE,
    SILENT_MODE,
    CALIBRATION_MODE
};

enum DeviceState {
    IDLE,
    LISTENING,
    PROCESSING,
    SPEAKING,
    SOS_ACTIVE,
    ERROR_STATE
};

// ===== CONFIGURATION STRUCTURES =====
struct DeviceConfig {
    float outputVolume = 0.7f;
    float noiseTreshold = 0.1f;
    bool autoMode = true;
    bool sosEnabled = true;
    uint8_t mlModelVersion = 1;
    char userVoiceProfile[32] = "default";
};

struct VoiceProfile {
    char userId[32];
    char userName[64];
    float voiceCharacteristics[PERSONALIZATION_FEATURES_SIZE];
    float pitchRange[2];          // [min_pitch, max_pitch]
    float timbreFeatures[32];     // Timbre characteristics
    float speakingRate;           // Words per minute
    float emotionalTone;          // Emotional baseline
    bool isCalibrated;
    unsigned long lastUsed;
    float adaptationWeights[64];  // Adaptation weights
};

struct VoiceFeatures {
    float melSpectrogram[MEL_BINS][VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
    float pitch[VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
    float energy[VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
    float spectralCentroid[VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
    float spectralRolloff[VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
    float mfcc[13][VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
    float zcr[VOICE_FRAME_SIZE/VOICE_HOP_SIZE];
};

struct PersonalizationConfig {
    float voiceClarity = 0.8f;
    float naturalness = 0.9f;
    float emotionalExpression = 0.7f;
    float speechTempo = 1.0f;
    float volumeAdaptation = 0.8f;
    bool noiseReduction = true;
    bool realTimeAdaptation = true;
};

// ===== GLOBAL DECLARATIONS =====
extern DeviceMode currentMode;
extern DeviceState currentState;
extern float batteryVoltage;
extern float batteryPercent;
extern bool isWearing;
extern bool sosActive;
extern unsigned long lastActivityTime;
extern DeviceConfig config;

// Task Handles
extern TaskHandle_t captureTaskHandle;
extern TaskHandle_t processingTaskHandle;
extern TaskHandle_t playbackTaskHandle;
extern TaskHandle_t batteryTaskHandle;
extern TaskHandle_t proximityTaskHandle;
extern TaskHandle_t bleTaskHandle;

// Queue Handles
extern QueueHandle_t audioInputQueue;
extern QueueHandle_t audioOutputQueue;
extern QueueHandle_t mlProcessingQueue;
extern QueueHandle_t statusQueue;

// Semaphore Handles
extern SemaphoreHandle_t i2sMutex;
extern SemaphoreHandle_t displayMutex;
extern SemaphoreHandle_t spiffsMutex;

// ===== FUNCTION DECLARATIONS =====
// Utility functions
void logMessage(const char* tag, const char* message);
void updateDisplay(const char* line1, const char* line2 = "", const char* line3 = "", const char* line4 = "");

// Configuration functions
void loadConfiguration();
void saveConfiguration();

// Power management
void enterLightSleep();
void handleOTAUpdate();

// System monitoring
void monitorTaskHealth();
void performSystemDiagnostics();
void emergency_shutdown();

// Hardware functions
void setVibroMotors(float intensity1, float intensity2);
void stopVibroMotors();

// Interrupt handlers
void IRAM_ATTR sos_button_handler();

#endif // VOICEX_CONFIG_H