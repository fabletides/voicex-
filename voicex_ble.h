// ===== VOICEX_BLE.H - BLUETOOTH MODULE =====
#ifndef VOICEX_BLE_H
#define VOICEX_BLE_H

#include "voicex_config.h"

// ===== BLE OBJECTS =====
extern BLEServer* pServer;
extern BLEService* pService;
extern BLECharacteristic* pAudioCharacteristic;
extern BLECharacteristic* pStatusCharacteristic;
extern BLECharacteristic* pControlCharacteristic;
extern bool deviceConnected;

// ===== FUNCTION DECLARATIONS =====
void initializeBLE();
void sendBatteryStatusViaBLE(float percent, float voltage, float current);
void sendPeriodicBLEStatus();
void sendPerformanceDataViaBLE(size_t freeHeap, size_t freeSPIRAM);
void handleBLEPersonalizationCommand(String cmdJson);

// BLE Callback classes
class MyServerCallbacks;
class EnhancedControlCallbacks;

#endif // VOICEX_BLE_H

// ===== VOICEX_BLE.CPP =====
#include "voicex_ble.h"
#include "voicex_personalization.h"

// ===== BLE OBJECTS =====
BLEServer* pServer = nullptr;
BLEService* pService = nullptr;
BLECharacteristic* pAudioCharacteristic = nullptr;
BLECharacteristic* pStatusCharacteristic = nullptr;
BLECharacteristic* pControlCharacteristic = nullptr;
bool deviceConnected = false;

// ===== BLE CALLBACKS =====
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        logMessage("BLE", "Device connected");
        updateDisplay("BLE Connected", "Enhanced Ready", "", "");
    }
    
    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        logMessage("BLE", "Device disconnected");
        updateDisplay("BLE Disconnected", "Searching...", "", "");
        BLEDevice::startAdvertising();
    }
};

class EnhancedControlCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pCharacteristic) {
        std::string value = pCharacteristic->getValue();
        if (value.length() > 0) {
            String command = String(value.c_str());
            
            // Check if it's a personalization command
            if (command.indexOf("action") >= 0) {
                handleBLEPersonalizationCommand(command);
            } else {
                // Handle regular commands
                handleRegularBLECommand(command);
            }
        }
    }
};

// ===== BLE INITIALIZATION =====
void initializeBLE() {
    logMessage("BLE", "Initializing enhanced BLE...");
    
    BLEDevice::init(BLE_DEVICE_NAME);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    pService = pServer->createService(BLE_SERVICE_UUID);
    
    // Audio characteristic for streaming
    pAudioCharacteristic = pService->createCharacteristic(
        BLE_CHAR_AUDIO_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    
    // Status characteristic for device status
    pStatusCharacteristic = pService->createCharacteristic(
        BLE_CHAR_STATUS_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    
    // Control characteristic for commands
    pControlCharacteristic = pService->createCharacteristic(
        BLE_CHAR_CONTROL_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );
    pControlCharacteristic->setCallbacks(new EnhancedControlCallbacks());
    
    pService->start();
    
    // Setup advertising
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(BLE_SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0);
    
    BLEDevice::startAdvertising();
    logMessage("BLE", "Enhanced BLE initialized and advertising");
}

// ===== BLE STATUS FUNCTIONS =====
void sendBatteryStatusViaBLE(float percent, float voltage, float current) {
    if (deviceConnected && pStatusCharacteristic) {
        StaticJsonDocument<256> status;
        status["type"] = "battery";
        status["battery_percent"] = percent;
        status["battery_voltage"] = voltage;
        status["current"] = current;
        status["timestamp"] = millis();
        
        String statusStr;
        serializeJson(status, statusStr);
        pStatusCharacteristic->setValue(statusStr.c_str());
        pStatusCharacteristic->notify();
    }
}

void sendPeriodicBLEStatus() {
    if (deviceConnected && pStatusCharacteristic) {
        StaticJsonDocument<512> status;
        status["type"] = "status";
        status["battery_percent"] = batteryPercent;
        status["battery_voltage"] = batteryVoltage;
        status["state"] = currentState;
        status["mode"] = currentMode;
        status["wearing"] = isWearing;
        status["sos_active"] = sosActive;
        status["personalization_active"] = isPersonalizationActive;
        status["current_profile"] = getCurrentProfileIndex();
        status["active_profiles"] = getActiveProfilesCount();
        status["uptime"] = millis();
        status["free_heap"] = ESP.getFreeHeap();
        status["voice_quality"] = calculateVoiceQualityScore();
        
        String statusStr;
        serializeJson(status, statusStr);
        pStatusCharacteristic->setValue(statusStr.c_str());
        pStatusCharacteristic->notify();
    }
}

void sendPerformanceDataViaBLE(size_t freeHeap, size_t freeSPIRAM) {
    if (deviceConnected && pStatusCharacteristic) {
        StaticJsonDocument<256> perfData;
        perfData["type"] = "performance";
        perfData["free_heap"] = freeHeap;
        perfData["free_psram"] = freeSPIRAM;
        perfData["cpu_usage"] = estimateCPUUsage();
        perfData["avg_latency"] = getAverageProcessingLatency();
        perfData["voice_quality"] = calculateVoiceQualityScore();
        
        String perfStr;
        serializeJson(perfData, perfStr);
        pStatusCharacteristic->setValue(perfStr.c_str());
        pStatusCharacteristic->notify();
    }
}

// ===== BLE COMMAND HANDLERS =====
void handleRegularBLECommand(String cmdJson) {
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, cmdJson);
    if (error) {
        logMessage("BLE", "Failed to parse JSON command");
        return;
    }
    
    if (doc.containsKey("mode")) {
        int mode = doc["mode"];
        currentMode = (DeviceMode)mode;
        logMessage("BLE", ("Mode changed to: " + String(mode)).c_str());
    }
    
    if (doc.containsKey("volume")) {
        float volume = doc["volume"];
        config.outputVolume = constrain(volume, 0.0f, 1.0f);
        logMessage("BLE", ("Volume changed to: " + String(config.outputVolume)).c_str());
    }
    
    if (doc.containsKey("sos")) {
        if (doc["sos"] == true) {
            sosActive = true;
            currentState = SOS_ACTIVE;
            logMessage("BLE", "SOS activated via BLE");
        }
    }
    
    if (doc.containsKey("calibrate")) {
        if (doc["calibrate"] == true) {
            calibrateDevice();
        }
    }
    
    if (doc.containsKey("diagnostics")) {
        if (doc["diagnostics"] == true) {
            performSystemDiagnostics();
        }
    }
}

void handleBLEPersonalizationCommand(String cmdJson) {
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, cmdJson);
    if (error) return;
    
    if (doc.containsKey("action")) {
        String action = doc["action"];
        
        if (action == "start_calibration") {
            int profileIndex = doc["profile_index"] | 0;
            String userName = doc["user_name"] | "User";
            String userId = doc["user_id"] | "default";
            
            startVoiceCalibration(profileIndex, userName, userId);
        }
        else if (action == "select_profile") {
            int profileIndex = doc["profile_index"] | 0;
            selectVoiceProfile(profileIndex);
        }
        else if (action == "update_config") {
            updatePersonalizationConfig(doc);
        }
        else if (action == "get_profiles") {
            sendVoiceProfilesList();
        }
        else if (action == "delete_profile") {
            int profileIndex = doc["profile_index"] | 0;
            deleteVoiceProfile(profileIndex);
        }
        else if (action == "export_profile") {
            int profileIndex = doc["profile_index"] | 0;
            exportVoiceProfile(profileIndex);
        }
        else if (action == "import_profile") {
            importVoiceProfile(doc);
        }
    }
}

// ===== VOICE PROFILES BLE FUNCTIONS =====
void sendVoiceProfilesList() {
    if (!deviceConnected || !pStatusCharacteristic) return;
    
    StaticJsonDocument<1024> doc;
    doc["type"] = "profiles_list";
    
    JsonArray profiles = doc.createNestedArray("profiles");
    
    for (int i = 0; i < MAX_VOICE_PROFILES; i++) {
        VoiceProfile* profile = getVoiceProfile(i);
        if (profile && profile->isCalibrated) {
            JsonObject profileObj = profiles.createNestedObject();
            profileObj["index"] = i;
            profileObj["user_name"] = profile->userName;
            profileObj["user_id"] = profile->userId;
            profileObj["last_used"] = profile->lastUsed;
            profileObj["speaking_rate"] = profile->speakingRate;
            profileObj["pitch_range_min"] = profile->pitchRange[0];
            profileObj["pitch_range_max"] = profile->pitchRange[1];
            profileObj["emotional_tone"] = profile->emotionalTone;
            profileObj["is_active"] = (i == getCurrentProfileIndex());
        }
    }
    
    String response;
    serializeJson(doc, response);
    pStatusCharacteristic->setValue(response.c_str());
    pStatusCharacteristic->notify();
}

void sendVoiceCalibrationProgress(int profileIndex, int currentSample, int totalSamples) {
    if (!deviceConnected || !pStatusCharacteristic) return;
    
    StaticJsonDocument<256> doc;
    doc["type"] = "calibration_progress";
    doc["profile_index"] = profileIndex;
    doc["current_sample"] = currentSample;
    doc["total_samples"] = totalSamples;
    doc["progress_percent"] = (currentSample * 100) / totalSamples;
    
    String response;
    serializeJson(doc, response);
    pStatusCharacteristic->setValue(response.c_str());
    pStatusCharacteristic->notify();
}

void sendVoiceCalibrationComplete(int profileIndex, bool success) {
    if (!deviceConnected || !pStatusCharacteristic) return;
    
    StaticJsonDocument<256> doc;
    doc["type"] = "calibration_complete";
    doc["profile_index"] = profileIndex;
    doc["success"] = success;
    doc["timestamp"] = millis();
    
    if (success) {
        VoiceProfile* profile = getVoiceProfile(profileIndex);
        if (profile) {
            doc["user_name"] = profile->userName;
            doc["pitch_range_min"] = profile->pitchRange[0];
            doc["pitch_range_max"] = profile->pitchRange[1];
            doc["speaking_rate"] = profile->speakingRate;
        }
    }
    
    String response;
    serializeJson(doc, response);
    pStatusCharacteristic->setValue(response.c_str());
    pStatusCharacteristic->notify();
}

// ===== AUDIO STREAMING VIA BLE =====
void sendAudioDataViaBLE(int16_t* audio_data, size_t size) {
    if (deviceConnected && pAudioCharacteristic) {
        // Send audio data in chunks to prevent BLE overflow
        const size_t chunk_size = 512;
        for (size_t i = 0; i < size; i += chunk_size) {
            size_t current_chunk = min(chunk_size, size - i);
            
            std::string audioStr((char*)&audio_data[i], current_chunk * sizeof(int16_t));
            pAudioCharacteristic->setValue(audioStr);
            pAudioCharacteristic->notify();
            
            vTaskDelay(pdMS_TO_TICKS(10)); // Prevent BLE overflow
        }
    }
}

// ===== UTILITY FUNCTIONS =====
float estimateCPUUsage() {
    static unsigned long lastCheck = 0;
    static unsigned long lastIdle = 0;
    
    unsigned long currentTime = millis();
    unsigned long currentIdle = esp_timer_get_time() / 1000; // Convert to ms
    
    if (lastCheck > 0) {
        unsigned long totalTime = currentTime - lastCheck;
        unsigned long idleTime = currentIdle - lastIdle;
        float cpuUsage = 100.0f - ((float)idleTime / totalTime * 100.0f);
        
        lastCheck = currentTime;
        lastIdle = currentIdle;
        
        return constrain(cpuUsage, 0.0f, 100.0f);
    }
    
    lastCheck = currentTime;
    lastIdle = currentIdle;
    return 0.0f;
}

float getAverageProcessingLatency() {
    static float latencyBuffer[10] = {0};
    static int bufferIndex = 0;
    static unsigned long lastProcessingTime = 0;
    
    unsigned long currentTime = millis();
    
    if (lastProcessingTime > 0) {
        float latency = currentTime - lastProcessingTime;
        latencyBuffer[bufferIndex] = latency;
        bufferIndex = (bufferIndex + 1) % 10;
    }
    
    lastProcessingTime = currentTime;
    
    // Calculate average
    float sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += latencyBuffer[i];
    }
    
    return sum / 10.0f;
}

float calculateVoiceQualityScore() {
    float qualityScore = 0.0f;
    
    // Base score
    qualityScore += 20.0f;
    
    // Personalization bonus
    if (isPersonalizationActive && getCurrentProfileIndex() >= 0) {
        qualityScore += 30.0f;
    }
    
    // Configuration quality
    PersonalizationConfig* config = getPersonalizationConfig();
    if (config) {
        qualityScore += config->voiceClarity * 25.0f;
        qualityScore += config->naturalness * 25.0f;
    }
    
    return min(qualityScore, 100.0f);
}

// ===== ERROR HANDLING =====
void sendErrorViaBLE(const char* errorType, const char* errorMessage) {
    if (!deviceConnected || !pStatusCharacteristic) return;
    
    StaticJsonDocument<256> doc;
    doc["type"] = "error";
    doc["error_type"] = errorType;
    doc["error_message"] = errorMessage;
    doc["timestamp"] = millis();
    
    String response;
    serializeJson(doc, response);
    pStatusCharacteristic->setValue(response.c_str());
    pStatusCharacteristic->notify();
}

void sendWarningViaBLE(const char* warningType, const char* warningMessage) {
    if (!deviceConnected || !pStatusCharacteristic) return;
    
    StaticJsonDocument<256> doc;
    doc["type"] = "warning";
    doc["warning_type"] = warningType;
    doc["warning_message"] = warningMessage;
    doc["timestamp"] = millis();
    
    String response;
    serializeJson(doc, response);
    pStatusCharacteristic->setValue(response.c_str());
    pStatusCharacteristic->notify();
}