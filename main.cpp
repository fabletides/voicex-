// ===== MAIN.CPP - ОСНОВНОЙ ФАЙЛ VOICEX =====
#include <voicex_config.h>
#include <voicex_hardware.h>
#include <voicex_ble.h>
#include <voicex_ml.h>
#include <voicex_audio.h>
#include <voicex_personalization.h>
#include <voicex_tasks.h>

// ===== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =====
// Device Status
DeviceMode currentMode = AUTO_MODE;
DeviceState currentState = IDLE;
float batteryVoltage = 0.0;
float batteryPercent = 0.0;
bool isWearing = false;
bool sosActive = false;
unsigned long lastActivityTime = 0;

// Configuration
DeviceConfig config;

// FreeRTOS Handles
TaskHandle_t captureTaskHandle = nullptr;
TaskHandle_t processingTaskHandle = nullptr;
TaskHandle_t playbackTaskHandle = nullptr;
TaskHandle_t batteryTaskHandle = nullptr;
TaskHandle_t proximityTaskHandle = nullptr;
TaskHandle_t bleTaskHandle = nullptr;

// Queue Handles
QueueHandle_t audioInputQueue;
QueueHandle_t audioOutputQueue;
QueueHandle_t mlProcessingQueue;
QueueHandle_t statusQueue;

// Semaphore Handles
SemaphoreHandle_t i2sMutex;
SemaphoreHandle_t displayMutex;
SemaphoreHandle_t spiffsMutex;

// ===== SETUP FUNCTION =====
void setup() {
    Serial.begin(115200);
    logMessage("MAIN", "VoiceX Enhanced starting up...");
    
    // Initialize hardware
    initializeHardware();
    
    // Initialize ML and personalization systems
    initializeMLSystem();
    initializePersonalizationSystem();
    
    // Initialize BLE
    initializeBLE();
    
    // Create mutexes
    i2sMutex = xSemaphoreCreateMutex();
    displayMutex = xSemaphoreCreateMutex();
    spiffsMutex = xSemaphoreCreateMutex();
    
    // Create queues
    audioInputQueue = xQueueCreate(10, I2S_BUFFER_SIZE * sizeof(int16_t));
    audioOutputQueue = xQueueCreate(10, I2S_BUFFER_SIZE * sizeof(int16_t));
    mlProcessingQueue = xQueueCreate(5, ML_INPUT_SIZE * sizeof(float));
    statusQueue = xQueueCreate(5, sizeof(DeviceState));
    
    if (!audioInputQueue || !audioOutputQueue || !mlProcessingQueue || !statusQueue) {
        logMessage("ERROR", "Failed to create queues");
        return;
    }
    
    // Setup SOS button interrupt
    attachInterrupt(digitalPinToInterrupt(SOS_BUTTON_PIN), sos_button_handler, CHANGE);
    
    // Create all tasks
    createAllTasks();
    
    // Initialize device state
    currentMode = AUTO_MODE;
    currentState = IDLE;
    lastActivityTime = millis();
    
    // Load configuration
    loadConfiguration();
    
    // Initialize status LED
    digitalWrite(STATUS_LED_PIN, LOW);
    
    logMessage("MAIN", "VoiceX Enhanced initialization complete");
    updateDisplay("VoiceX Enhanced", "ML Ready", "Personalization: ON", "BLE: Advertising");
}

// ===== MAIN LOOP =====
void loop() {
    static unsigned long lastHeartbeat = 0;
    static unsigned long lastConfigSave = 0;
    
    unsigned long currentTime = millis();
    
    // Heartbeat every 10 seconds
    if (currentTime - lastHeartbeat > 10000) {
        logMessage("HEARTBEAT", "System running");
        lastHeartbeat = currentTime;
        
        // Check memory
        size_t freeHeap = ESP.getFreeHeap();
        if (freeHeap < 20000) {
            logMessage("WARNING", ("Low memory: " + String(freeHeap) + " bytes").c_str());
        }
        
        // Monitor task health
        monitorTaskHealth();
    }
    
    // Save configuration every 5 minutes
    if (currentTime - lastConfigSave > 300000) {
        saveConfiguration();
        lastConfigSave = currentTime;
    }
    
    // Handle power management
    if (currentState == IDLE && currentTime - lastActivityTime > 30000) {
        enterLightSleep();
    }
    
    // Handle OTA updates
    handleOTAUpdate();
    
    // Reset watchdog
    esp_task_wdt_reset();
    
    vTaskDelay(pdMS_TO_TICKS(100));
}

// ===== UTILITY FUNCTIONS =====
void logMessage(const char* tag, const char* message) {
    Serial.printf("[%s] %s\n", tag, message);
}

void monitorTaskHealth() {
    static unsigned long lastCheck = 0;
    
    if (millis() - lastCheck > 30000) {
        lastCheck = millis();
        
        // Check critical tasks
        if (captureTaskHandle && eTaskGetState(captureTaskHandle) == eDeleted) {
            logMessage("ERROR", "Audio capture task died - restarting");
            xTaskCreatePinnedToCore(capture_audio_task, "AudioCapture", 4096, NULL, 2, &captureTaskHandle, 0);
        }
        
        if (processingTaskHandle && eTaskGetState(processingTaskHandle) == eDeleted) {
            logMessage("ERROR", "Processing task died - restarting");
            xTaskCreatePinnedToCore(enhanced_voice_processing_task, "VoiceProcessing", 16384, NULL, 4, &processingTaskHandle, 1);
        }
        
        if (playbackTaskHandle && eTaskGetState(playbackTaskHandle) == eDeleted) {
            logMessage("ERROR", "Playback task died - restarting");
            xTaskCreatePinnedToCore(audio_playback_task, "AudioPlayback", 4096, NULL, 2, &playbackTaskHandle, 0);
        }
    }
}

// ===== INTERRUPT HANDLERS =====
void IRAM_ATTR sos_button_handler() {
    static unsigned long lastInterrupt = 0;
    unsigned long currentTime = millis();
    
    if (currentTime - lastInterrupt < 50) return;
    lastInterrupt = currentTime;
    
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    DeviceState newState = SOS_ACTIVE;
    xQueueSendFromISR(statusQueue, &newState, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}