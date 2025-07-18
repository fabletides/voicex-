// ===== VOICEX_TASKS.H - ЗАДАЧИ FREERTOS =====
#ifndef VOICEX_TASKS_H
#define VOICEX_TASKS_H

#include "voicex_config.h"

// ===== FUNCTION DECLARATIONS =====
void createAllTasks();

// Task functions
void capture_audio_task(void* parameter);
void enhanced_voice_processing_task(void* parameter);
void audio_playback_task(void* parameter);
void battery_monitor_task(void* parameter);
void proximity_activation_task(void* parameter);
void ble_server_handler_task(void* parameter);
void performance_monitoring_task(void* parameter);

#endif // VOICEX_TASKS_H

// ===== VOICEX_TASKS.CPP =====
#include "voicex_tasks.h"
#include "voicex_hardware.h"
#include "voicex_personalization.h"
#include "voicex_ble.h"

// ===== CREATE ALL TASKS =====
void createAllTasks() {
    logMessage("TASKS", "Creating all FreeRTOS tasks...");
    
    // Audio capture task (Core 0)
    xTaskCreatePinnedToCore(
        capture_audio_task,
        "AudioCapture",
        4096,
        NULL,
        2,
        &captureTaskHandle,
        0
    );
    
    // Enhanced voice processing task (Core 1)
    xTaskCreatePinnedToCore(
        enhanced_voice_processing_task,
        "VoiceProcessing",
        16384,
        NULL,
        4,
        &processingTaskHandle,
        1
    );
    
    // Audio playback task (Core 0)
    xTaskCreatePinnedToCore(
        audio_playback_task,
        "AudioPlayback",
        4096,
        NULL,
        2,
        &playbackTaskHandle,
        0
    );
    
    // Battery monitor task (Core 0)
    xTaskCreatePinnedToCore(
        battery_monitor_task,
        "BatteryMonitor",
        2048,
        NULL,
        1,
        &batteryTaskHandle,
        0
    );
    
    // Proximity activation task (Core 0)
    xTaskCreatePinnedToCore(
        proximity_activation_task,
        "ProximityActivation",
        2048,
        NULL,
        1,
        &proximityTaskHandle,
        0
    );
    
    // BLE handler task (Core 0)
    xTaskCreatePinnedToCore(
        ble_server_handler_task,
        "BLEHandler",
        4096,
        NULL,
        1,
        &bleTaskHandle,
        0
    );
    
    // Performance monitoring task (Core 0)
    xTaskCreatePinnedToCore(
        performance_monitoring_task,
        "PerformanceMonitor",
        4096,
        NULL,
        1,
        NULL,
        0
    );
    
    logMessage("TASKS", "All tasks created successfully");
}

// ===== AUDIO CAPTURE TASK =====
void capture_audio_task(void* parameter) {
    logMessage("TASK", "Audio capture task started");
    
    size_t bytes_read = 0;
    
    while (true) {
        if (currentState == LISTENING || currentState == PROCESSING) {
            if (xSemaphoreTake(i2sMutex, portMAX_DELAY)) {
                esp_err_t result = i2s_read(I2S_NUM_0, audio_input_buffer, 
                                          I2S_BUFFER_SIZE * sizeof(int16_t), 
                                          &bytes_read, portMAX_DELAY);
                
                if (result == ESP_OK && bytes_read > 0) {
                    // Send to processing queue
                    if (xQueueSend(audioInputQueue, audio_input_buffer, 0) != pdTRUE) {
                        logMessage("WARNING", "Audio input queue full");
                    }
                    
                    // Voice activity detection
                    float energy = 0.0f;
                    for (int i = 0; i < bytes_read / sizeof(int16_t); i++) {
                        energy += abs(audio_input_buffer[i]);
                    }
                    energy /= (bytes_read / sizeof(int16_t));
                    
                    if (energy > config.noiseTreshold * 32767 && currentState == LISTENING) {
                        currentState = PROCESSING;
                        lastActivityTime = millis();
                        logMessage("AUDIO", "Voice activity detected");
                    }
                }
                
                xSemaphoreGive(i2sMutex);
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ===== ENHANCED VOICE PROCESSING TASK =====
void enhanced_voice_processing_task(void* parameter) {
    logMessage("TASK", "Enhanced voice processing task started");
    
    int16_t* receivedBuffer = (int16_t*)malloc(I2S_BUFFER_SIZE * sizeof(int16_t));
    float* processedAudio = (float*)heap_caps_malloc(ML_OUTPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!receivedBuffer || !processedAudio) {
        logMessage("ERROR", "Failed to allocate processing buffers");
        vTaskDelete(NULL);
        return;
    }
    
    while (true) {
        if (xQueueReceive(audioInputQueue, receivedBuffer, portMAX_DELAY)) {
            unsigned long processingStart = millis();
            
            // Convert to float and normalize
            for (int i = 0; i < min(I2S_BUFFER_SIZE, ML_INPUT_SIZE); i++) {
                ml_input_buffer[i] = (float)receivedBuffer[i] / 32767.0f;
            }
            
            // Enhanced voice processing
            bool success = processVoiceWithPersonalization(ml_input_buffer, processedAudio);
            
            if (success) {
                // Convert back to audio samples
                for (int i = 0; i < min(ML_OUTPUT_SIZE, I2S_BUFFER_SIZE); i++) {
                    audio_output_buffer[i] = (int16_t)(processedAudio[i] * 32767.0f * config.outputVolume);
                }
                
                // Send to playback queue
                if (xQueueSend(audioOutputQueue, audio_output_buffer, 0) != pdTRUE) {
                    logMessage("WARNING", "Audio output queue full");
                }
                
                currentState = SPEAKING;
            } else {
                logMessage("ERROR", "Voice processing failed");
                currentState = LISTENING;
            }
            
            unsigned long processingTime = millis() - processingStart;
            if (processingTime > 100) {
                logMessage("PERFORMANCE", ("Processing time: " + String(processingTime) + "ms").c_str());
            }
        }
    }
    
    free(receivedBuffer);
    free(processedAudio);
    vTaskDelete(NULL);
}

// ===== AUDIO PLAYBACK TASK =====
void audio_playback_task(void* parameter) {
    logMessage("TASK", "Audio playback task started");
    
    int16_t* playbackBuffer = (int16_t*)malloc(I2S_BUFFER_SIZE * sizeof(int16_t));
    size_t bytes_written = 0;
    
    if (!playbackBuffer) {
        logMessage("ERROR", "Failed to allocate playback buffer");
        vTaskDelete(NULL);
        return;
    }
    
    while (true) {
        if (xQueueReceive(audioOutputQueue, playbackBuffer, portMAX_DELAY)) {
            if (currentState == SPEAKING) {
                // Choose output method based on mode
                if (currentMode == LOUD_MODE) {
                    playAudioViaDAC(playbackBuffer, I2S_BUFFER_SIZE);
                } else {
                    if (xSemaphoreTake(i2sMutex, portMAX_DELAY)) {
                        i2s_write(I2S_NUM_1, playbackBuffer, 
                                 I2S_BUFFER_SIZE * sizeof(int16_t), 
                                 &bytes_written, portMAX_DELAY);
                        xSemaphoreGive(i2sMutex);
                    }
                }
                
                // Control vibration motors based on audio
                float vibro_intensity = 0.0f;
                for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
                    vibro_intensity += abs(playbackBuffer[i]);
                }
                vibro_intensity /= (I2S_BUFFER_SIZE * 32767.0f);
                
                // Dual vibration with phase difference for naturalness
                setVibroMotors(vibro_intensity, vibro_intensity * 0.8f);
                
                // Check if speaking finished
                if (vibro_intensity < 0.01f) {
                    currentState = LISTENING;
                    stopVibroMotors();
                    logMessage("AUDIO", "Speech completed");
                }
            }
        }
    }
    
    free(playbackBuffer);
    vTaskDelete(NULL);
}

// ===== BATTERY MONITOR TASK =====
void battery_monitor_task(void* parameter) {
    logMessage("TASK", "Battery monitor task started");
    
    while (true) {
        // Read battery data
        batteryVoltage = ina219.getBusVoltage_V();
        float current = ina219.getCurrent_mA();
        
        // Calculate battery percentage (3.7V Li-ion)
        if (batteryVoltage > 4.2f) {
            batteryPercent = 100.0f;
        } else if (batteryVoltage > 3.0f) {
            batteryPercent = ((batteryVoltage - 3.0f) / 1.2f) * 100.0f;
        } else {
            batteryPercent = 0.0f;
        }
        
        // Send status to BLE
        sendBatteryStatusViaBLE(batteryPercent, batteryVoltage, current);
        
        // Update display
        char line1[32], line2[32], line3[32], line4[32];
        sprintf(line1, "Battery: %.1f%%", batteryPercent);
        sprintf(line2, "State: %s", getStateString(currentState));
        sprintf(line3, "Mode: %s", getModeString(currentMode));
        sprintf(line4, "Profiles: %d", getActiveProfilesCount());
        
        updateDisplay(line1, line2, line3, line4);
        
        // Check battery level
        checkBatteryLevel();
        
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

// ===== PROXIMITY ACTIVATION TASK =====
void proximity_activation_task(void* parameter) {
    logMessage("TASK", "Proximity activation task started");
    
    while (true) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        
        // Wearing detection based on accelerometer
        float total_acceleration = sqrt(a.acceleration.x * a.acceleration.x + 
                                       a.acceleration.y * a.acceleration.y + 
                                       a.acceleration.z * a.acceleration.z);
        
        static float prev_acc = 0.0f;
        float acc_diff = abs(total_acceleration - prev_acc);
        prev_acc = total_acceleration;
        
        // Device is worn if acceleration is close to gravity and stable
        isWearing = (total_acceleration > 8.0f && total_acceleration < 12.0f && acc_diff < 2.0f);
        
        // Auto mode activation
        if (currentMode == AUTO_MODE) {
            if (isWearing && currentState == IDLE) {
                currentState = LISTENING;
                digitalWrite(STATUS_LED_PIN, HIGH);
                logMessage("AUTO", "Device activated - wearing detected");
            } else if (!isWearing && currentState == LISTENING) {
                currentState = IDLE;
                digitalWrite(STATUS_LED_PIN, LOW);
                logMessage("AUTO", "Device deactivated - not wearing");
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// ===== BLE SERVER HANDLER TASK =====
void ble_server_handler_task(void* parameter) {
    logMessage("TASK", "BLE server handler task started");
    
    while (true) {
        // Handle BLE connection status and periodic updates
        if (deviceConnected) {
            // Send periodic status updates
            sendPeriodicBLEStatus();
            vTaskDelay(pdMS_TO_TICKS(2000));
        } else {
            // Check for reconnection
            vTaskDelay(pdMS_TO_TICKS(5000));
        }
    }
}

// ===== PERFORMANCE MONITORING TASK =====
void performance_monitoring_task(void* parameter) {
    logMessage("TASK", "Performance monitoring task started");
    
    while (true) {
        static unsigned long lastReport = 0;
        unsigned long currentTime = millis();
        
        if (currentTime - lastReport > 30000) { // Report every 30 seconds
            lastReport = currentTime;
            
            // Memory monitoring
            size_t freeHeap = ESP.getFreeHeap();
            size_t freeSPIRAM = ESP.getFreePsram();
            
            // Task monitoring
            monitorTaskHealth();
            
            // Send performance data via BLE
            sendPerformanceDataViaBLE(freeHeap, freeSPIRAM);
            
            logMessage("PERFORMANCE", ("Free Heap: " + String(freeHeap) + 
                                     " PSRAM: " + String(freeSPIRAM)).c_str());
            
            // Check for low memory
            if (freeHeap < 10000) {
                logMessage("WARNING", "Low memory detected");
                // Trigger cleanup if needed
            }
        }
        
        // Reset watchdog
        esp_task_wdt_reset();
        
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}

// ===== UTILITY FUNCTIONS FOR TASKS =====
const char* getStateString(DeviceState state) {
    switch (state) {
        case IDLE: return "IDLE";
        case LISTENING: return "LISTENING";
        case PROCESSING: return "PROCESSING";
        case SPEAKING: return "SPEAKING";
        case SOS_ACTIVE: return "SOS";
        case ERROR_STATE: return "ERROR";
        default: return "UNKNOWN";
    }
}

const char* getModeString(DeviceMode mode) {
    switch (mode) {
        case AUTO_MODE: return "AUTO";
        case MANUAL_MODE: return "MANUAL";
        case LOUD_MODE: return "LOUD";
        case SOS_MODE: return "SOS";
        case SILENT_MODE: return "SILENT";
        case CALIBRATION_MODE: return "CALIBRATION";
        default: return "UNKNOWN";
    }
}

// DAC Audio playback function
void playAudioViaDAC(int16_t* audio_data, size_t size) {
    dac_output_enable(DAC_CHANNEL_1);
    
    for (size_t i = 0; i < size; i++) {
        uint8_t dac_value = (audio_data[i] + 32768) >> 8; // Convert to 8-bit DAC
        dac_output_voltage(DAC_CHANNEL_1, dac_value);
        delayMicroseconds(1000000 / I2S_SAMPLE_RATE); // Sample rate timing
    }
    
    dac_output_disable(DAC_CHANNEL_1);
}