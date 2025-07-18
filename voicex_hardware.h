// ===== VOICEX_HARDWARE.H - АППАРАТНАЯ ЧАСТЬ =====
#ifndef VOICEX_HARDWARE_H
#define VOICEX_HARDWARE_H

#include "voicex_config.h"

// ===== HARDWARE OBJECTS =====
extern Adafruit_SSD1306 display;
extern Adafruit_MPU6050 mpu;
extern Adafruit_INA219 ina219;

// ===== AUDIO BUFFERS =====
extern int16_t* audio_input_buffer;
extern int16_t* audio_output_buffer;
extern float* ml_input_buffer;
extern float* ml_output_buffer;

// ===== FUNCTION DECLARATIONS =====
void initializeHardware();
void setupPeripherals();
void setupI2S();
void setupDisplay();
void setupSensors();
void calibrateDevice();

// Display functions
void updateDisplay(const char* line1, const char* line2, const char* line3, const char* line4);

// Motor control
void setVibroMotors(float intensity1, float intensity2);
void stopVibroMotors();

// Configuration
void loadConfiguration();
void saveConfiguration();

// Power management
void enterLightSleep();
void setupPowerManagement();
void checkBatteryLevel();

// OTA
void handleOTAUpdate();
void setupWiFi();
void setupOTA();

#endif // VOICEX_HARDWARE_H

// ===== VOICEX_HARDWARE.CPP =====
#include "voicex_hardware.h"

// ===== HARDWARE OBJECTS =====
Adafruit_SSD1306 display(128, 64, &Wire, -1);
Adafruit_MPU6050 mpu;
Adafruit_INA219 ina219;

// Audio buffers
int16_t* audio_input_buffer = nullptr;
int16_t* audio_output_buffer = nullptr;
float* ml_input_buffer = nullptr;
float* ml_output_buffer = nullptr;

// WiFi credentials
const char* wifi_ssid = "YOUR_WIFI_SSID";
const char* wifi_password = "YOUR_WIFI_PASSWORD";

// ===== HARDWARE INITIALIZATION =====
void initializeHardware() {
    logMessage("HARDWARE", "Initializing hardware...");
    
    setupPeripherals();
    setupI2S();
    setupDisplay();
    setupSensors();
    setupPowerManagement();
    setupWiFi();
    
    logMessage("HARDWARE", "Hardware initialization complete");
}

void setupPeripherals() {
    // Initialize GPIO
    pinMode(VIBRO_MOTOR_1_PIN, OUTPUT);
    pinMode(VIBRO_MOTOR_2_PIN, OUTPUT);
    pinMode(SOS_BUTTON_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);
    pinMode(PROXIMITY_SENSOR_PIN, INPUT);
    
    // Initialize I2C
    Wire.begin(OLED_SDA, OLED_SCL);
    
    // Initialize SPIFFS
    if (!SPIFFS.begin(true)) {
        logMessage("ERROR", "SPIFFS Mount Failed");
    }
    
    logMessage("HARDWARE", "Peripherals initialized");
}

void setupI2S() {
    // I2S configuration for input
    i2s_config_t i2s_config_in = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = I2S_BUFFER_COUNT,
        .dma_buf_len = I2S_BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config_in = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    
    i2s_driver_install(I2S_NUM_0, &i2s_config_in, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config_in);
    i2s_zero_dma_buffer(I2S_NUM_0);
    
    // I2S configuration for output
    i2s_config_t i2s_config_out = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = I2S_BUFFER_COUNT,
        .dma_buf_len = I2S_BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config_out = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_SD_OUT,
        .data_in_num = I2S_PIN_NO_CHANGE
    };
    
    i2s_driver_install(I2S_NUM_1, &i2s_config_out, 0, NULL);
    i2s_set_pin(I2S_NUM_1, &pin_config_out);
    i2s_zero_dma_buffer(I2S_NUM_1);
    
    // Allocate audio buffers
    audio_input_buffer = (int16_t*)heap_caps_malloc(I2S_BUFFER_SIZE * sizeof(int16_t), MALLOC_CAP_DMA);
    audio_output_buffer = (int16_t*)heap_caps_malloc(I2S_BUFFER_SIZE * sizeof(int16_t), MALLOC_CAP_DMA);
    ml_input_buffer = (float*)heap_caps_malloc(ML_INPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    ml_output_buffer = (float*)heap_caps_malloc(ML_OUTPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!audio_input_buffer || !audio_output_buffer || !ml_input_buffer || !ml_output_buffer) {
        logMessage("ERROR", "Failed to allocate audio buffers");
        emergency_shutdown();
        return;
    }
    
    logMessage("HARDWARE", "I2S and audio buffers initialized");
}

void setupDisplay() {
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        logMessage("ERROR", "SSD1306 allocation failed");
        return;
    }
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("VoiceX Enhanced");
    display.println("Initializing...");
    display.display();
    
    logMessage("HARDWARE", "Display initialized");
}

void setupSensors() {
    // Initialize MPU6050
    if (!mpu.begin()) {
        logMessage("ERROR", "Failed to find MPU6050");
    } else {
        mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
        logMessage("HARDWARE", "MPU6050 initialized");
    }
    
    // Initialize INA219
    if (!ina219.begin()) {
        logMessage("ERROR", "Failed to find INA219");
    } else {
        logMessage("HARDWARE", "INA219 initialized");
    }
}

// ===== DISPLAY FUNCTIONS =====
void updateDisplay(const char* line1, const char* line2, const char* line3, const char* line4) {
    if (xSemaphoreTake(displayMutex, portMAX_DELAY)) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        
        display.setCursor(0, 0);
        display.println(line1);
        display.setCursor(0, 16);
        display.println(line2);
        display.setCursor(0, 32);
        display.println(line3);
        display.setCursor(0, 48);
        display.println(line4);
        
        display.display();
        xSemaphoreGive(displayMutex);
    }
}

// ===== MOTOR CONTROL =====
void setVibroMotors(float intensity1, float intensity2) {
    int pwm1 = (int)(intensity1 * 255);
    int pwm2 = (int)(intensity2 * 255);
    
    analogWrite(VIBRO_MOTOR_1_PIN, pwm1);
    analogWrite(VIBRO_MOTOR_2_PIN, pwm2);
}

void stopVibroMotors() {
    analogWrite(VIBRO_MOTOR_1_PIN, 0);
    analogWrite(VIBRO_MOTOR_2_PIN, 0);
}

// ===== CONFIGURATION =====
void loadConfiguration() {
    logMessage("CONFIG", "Loading configuration from SPIFFS");
    
    if (xSemaphoreTake(spiffsMutex, portMAX_DELAY)) {
        if (SPIFFS.exists("/config.json")) {
            File configFile = SPIFFS.open("/config.json", "r");
            if (configFile) {
                StaticJsonDocument<512> doc;
                DeserializationError error = deserializeJson(doc, configFile);
                
                if (!error) {
                    config.outputVolume = doc["outputVolume"] | 0.7f;
                    config.noiseTreshold = doc["noiseTreshold"] | 0.1f;
                    config.autoMode = doc["autoMode"] | true;
                    config.sosEnabled = doc["sosEnabled"] | true;
                    config.mlModelVersion = doc["mlModelVersion"] | 1;
                    strlcpy(config.userVoiceProfile, doc["userVoiceProfile"] | "default", 
                           sizeof(config.userVoiceProfile));
                    
                    logMessage("CONFIG", "Configuration loaded successfully");
                } else {
                    logMessage("ERROR", "Failed to parse configuration JSON");
                }
                configFile.close();
            }
        } else {
            logMessage("CONFIG", "No configuration file found, using defaults");
            saveConfiguration();
        }
        xSemaphoreGive(spiffsMutex);
    }
}

void saveConfiguration() {
    logMessage("CONFIG", "Saving configuration to SPIFFS");
    
    if (xSemaphoreTake(spiffsMutex, portMAX_DELAY)) {
        File configFile = SPIFFS.open("/config.json", "w");
        if (configFile) {
            StaticJsonDocument<512> doc;
            doc["outputVolume"] = config.outputVolume;
            doc["noiseTreshold"] = config.noiseTreshold;
            doc["autoMode"] = config.autoMode;
            doc["sosEnabled"] = config.sosEnabled;
            doc["mlModelVersion"] = config.mlModelVersion;
            doc["userVoiceProfile"] = config.userVoiceProfile;
            
            serializeJson(doc, configFile);
            configFile.close();
            logMessage("CONFIG", "Configuration saved successfully");
        } else {
            logMessage("ERROR", "Failed to open config file for writing");
        }
        xSemaphoreGive(spiffsMutex);
    }
}

// ===== POWER MANAGEMENT =====
void setupPowerManagement() {
    esp_sleep_enable_ext0_wakeup(GPIO_NUM_13, 0); // SOS button
    esp_sleep_enable_timer_wakeup(30 * 1000000); // 30 seconds
    
    esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
    esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_SLOW_MEM, ESP_PD_OPTION_ON);
    esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_FAST_MEM, ESP_PD_OPTION_ON);
    
    esp_task_wdt_init(30, true);
    esp_task_wdt_add(NULL);
    
    logMessage("POWER", "Power management configured");
}

void enterLightSleep() {
    logMessage("POWER", "Entering light sleep mode");
    
    if (captureTaskHandle) vTaskSuspend(captureTaskHandle);
    if (processingTaskHandle) vTaskSuspend(processingTaskHandle);
    
    saveConfiguration();
    
    esp_light_sleep_start();
    
    if (captureTaskHandle) vTaskResume(captureTaskHandle);
    if (processingTaskHandle) vTaskResume(processingTaskHandle);
    
    logMessage("POWER", "Woke up from light sleep");
    lastActivityTime = millis();
}

void checkBatteryLevel() {
    if (batteryPercent < 10.0f) {
        logMessage("POWER", "Low battery warning");
        updateDisplay("LOW BATTERY", "< 10%", "Please charge", "");
        
        config.outputVolume *= 0.5f;
        
        if (batteryPercent < 5.0f) {
            logMessage("POWER", "Critical battery - entering deep sleep");
            esp_deep_sleep_start();
        }
    }
}

// ===== WiFi AND OTA =====
void setupWiFi() {
    if (strlen(wifi_ssid) > 0) {
        logMessage("WIFI", "Connecting to WiFi...");
        WiFi.mode(WIFI_STA);
        WiFi.begin(wifi_ssid, wifi_password);
        
        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
            delay(500);
            attempts++;
        }
        
        if (WiFi.status() == WL_CONNECTED) {
            logMessage("WIFI", ("Connected to WiFi: " + WiFi.localIP().toString()).c_str());
            updateDisplay("WiFi Connected", WiFi.localIP().toString().c_str(), "", "");
            setupOTA();
        } else {
            logMessage("WIFI", "Failed to connect to WiFi");
            updateDisplay("WiFi Failed", "Check credentials", "", "");
        }
    }
}

void setupOTA() {
    ArduinoOTA.setHostname("voicex-enhanced");
    ArduinoOTA.setPassword("voicex2024");
    
    ArduinoOTA.onStart([]() {
        String type = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
        logMessage("OTA", ("Start updating " + type).c_str());
        updateDisplay("OTA Update", "Starting...", type.c_str(), "");
    });
    
    ArduinoOTA.onEnd([]() {
        logMessage("OTA", "Update completed");
        updateDisplay("OTA Update", "Completed", "Restarting...", "");
    });
    
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        unsigned int percent = (progress / (total / 100));
        updateDisplay("OTA Update", ("Progress: " + String(percent) + "%").c_str(), "", "");
    });
    
    ArduinoOTA.onError([](ota_error_t error) {
        String errorMsg;
        switch (error) {
            case OTA_AUTH_ERROR: errorMsg = "Auth Failed"; break;
            case OTA_BEGIN_ERROR: errorMsg = "Begin Failed"; break;
            case OTA_CONNECT_ERROR: errorMsg = "Connect Failed"; break;
            case OTA_RECEIVE_ERROR: errorMsg = "Receive Failed"; break;
            case OTA_END_ERROR: errorMsg = "End Failed"; break;
            default: errorMsg = "Unknown Error"; break;
        }
        logMessage("OTA", ("Error: " + errorMsg).c_str());
        updateDisplay("OTA Error", errorMsg.c_str(), "", "");
    });
    
    ArduinoOTA.begin();
    logMessage("OTA", "OTA service started");
}

void handleOTAUpdate() {
    static bool otaInitialized = false;
    
    if (!otaInitialized && WiFi.status() == WL_CONNECTED) {
        setupOTA();
        otaInitialized = true;
    }
    
    if (otaInitialized && WiFi.status() == WL_CONNECTED) {
        ArduinoOTA.handle();
    }
}

// ===== DEVICE CALIBRATION =====
void calibrateDevice() {
    logMessage("CALIBRATION", "Starting device calibration");
    DeviceMode prevMode = currentMode;
    currentMode = CALIBRATION_MODE;
    
    updateDisplay("Calibration", "Starting...", "Please wait", "");
    
    // Calibrate accelerometer
    logMessage("CALIBRATION", "Calibrating accelerometer");
    updateDisplay("Calibration", "Accelerometer", "Keep still", "");
    
    float accel_offset[3] = {0, 0, 0};
    const int samples = 100;
    
    for (int i = 0; i < samples; i++) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        
        accel_offset[0] += a.acceleration.x;
        accel_offset[1] += a.acceleration.y;
        accel_offset[2] += a.acceleration.z;
        
        if (i % 10 == 0) {
            updateDisplay("Calibration", "Accelerometer", 
                         ("Progress: " + String(i) + "%").c_str(), "");
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    for (int i = 0; i < 3; i++) {
        accel_offset[i] /= samples;
    }
    
    logMessage("CALIBRATION", ("Accel offsets: X=" + String(accel_offset[0]) + 
                              " Y=" + String(accel_offset[1]) + 
                              " Z=" + String(accel_offset[2])).c_str());
    
    // Calibrate audio levels
    logMessage("CALIBRATION", "Calibrating noise threshold");
    updateDisplay("Calibration", "Noise Level", "Stay quiet", "");
    
    float noise_level = 0.0f;
    const int noise_samples = 50;
    
    for (int i = 0; i < noise_samples; i++) {
        size_t bytes_read = 0;
        esp_err_t result = i2s_read(I2S_NUM_0, audio_input_buffer, 
                                   I2S_BUFFER_SIZE * sizeof(int16_t), 
                                   &bytes_read, portMAX_DELAY);
        
        if (result == ESP_OK && bytes_read > 0) {
            float energy = 0.0f;
            for (int j = 0; j < bytes_read / sizeof(int16_t); j++) {
                energy += abs(audio_input_buffer[j]);
            }
            noise_level += energy / (bytes_read / sizeof(int16_t));
        }
        
        if (i % 5 == 0) {
            updateDisplay("Calibration", "Noise Level", 
                         ("Progress: " + String(i * 2) + "%").c_str(), "");
        }
        
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    
    config.noiseTreshold = (noise_level / noise_samples) / 32767.0f * 1.5f;
    logMessage("CALIBRATION", ("Noise threshold set to: " + String(config.noiseTreshold)).c_str());
    
    // Test vibration motors
    logMessage("CALIBRATION", "Testing vibration motors");
    updateDisplay("Calibration", "Vibration Test", "Feel the buzz", "");
    
    for (int i = 0; i < 3; i++) {
        setVibroMotors(1.0f, 0.0f);
        vTaskDelay(pdMS_TO_TICKS(300));
        setVibroMotors(0.0f, 1.0f);
        vTaskDelay(pdMS_TO_TICKS(300));
        setVibroMotors(1.0f, 1.0f);
        vTaskDelay(pdMS_TO_TICKS(300));
        stopVibroMotors();
        vTaskDelay(pdMS_TO_TICKS(200));
    }
    
    // Save calibration data
    saveConfiguration();
    
    currentMode = prevMode;
    updateDisplay("Calibration", "Completed", "Device ready", "");
    
    logMessage("CALIBRATION", "Device calibration completed");
    vTaskDelay(pdMS_TO_TICKS(2000));
}