// ===== VOICEX_PERSONALIZATION.H - ПЕРСОНАЛИЗАЦИЯ =====
#ifndef VOICEX_PERSONALIZATION_H
#define VOICEX_PERSONALIZATION_H

#include "voicex_config.h"

// ===== GLOBAL VARIABLES =====
extern VoiceProfile userProfiles[MAX_VOICE_PROFILES];
extern int currentProfileIndex;
extern PersonalizationConfig personalizationConfig;
extern bool isPersonalizationActive;
extern bool isVoiceCalibrationMode;

// Enhanced buffers
extern float* voiceInputBuffer;
extern float* voiceOutputBuffer;
extern float* personalizedOutputBuffer;
extern VoiceFeatures currentVoiceFeatures;

// ===== FUNCTION DECLARATIONS =====
// Initialization
void initializePersonalizationSystem();

// Voice calibration
void startVoiceCalibration(int profileIndex, String userName, String userId);
void voice_calibration_task(void* parameter);

// Voice processing
bool processVoiceWithPersonalization(float* inputAudio, float* outputAudio);
void extractVoiceFeatures(float* audioBuffer, int bufferSize, VoiceFeatures* features);

// Profile management
VoiceProfile* getVoiceProfile(int index);
int getCurrentProfileIndex();
int getActiveProfilesCount();
void selectVoiceProfile(int profileIndex);
void deleteVoiceProfile(int profileIndex);
void exportVoiceProfile(int profileIndex);
void importVoiceProfile(StaticJsonDocument<512>& doc);

// Configuration
PersonalizationConfig* getPersonalizationConfig();
void updatePersonalizationConfig(StaticJsonDocument<512>& doc);

// Real-time adaptation
void performRealTimeAdaptation(float* inputAudio, float* outputAudio, VoiceProfile* profile);

// Audio enhancement
void applyPersonalizationEffects(float* audioBuffer, int bufferSize, VoiceProfile* profile);

#endif // VOICEX_PERSONALIZATION_H

// ===== VOICEX_PERSONALIZATION.CPP =====
#include "voicex_personalization.h"
#include "voicex_ml.h"
#include "voicex_ble.h"

// ===== GLOBAL VARIABLES =====
VoiceProfile userProfiles[MAX_VOICE_PROFILES];
int currentProfileIndex = -1;
PersonalizationConfig personalizationConfig;
bool isPersonalizationActive = false;
bool isVoiceCalibrationMode = false;

// Enhanced audio buffers
float* voiceInputBuffer = nullptr;
float* voiceOutputBuffer = nullptr;
float* personalizedOutputBuffer = nullptr;
VoiceFeatures currentVoiceFeatures;

// Processing buffers
float* melSpectrogramBuffer = nullptr;
float* voiceFeaturesBuffer = nullptr;
float* pitchBuffer = nullptr;
float* energyBuffer = nullptr;

// ===== INITIALIZATION =====
void initializePersonalizationSystem() {
    logMessage("PERSONALIZATION", "Initializing personalization system...");
    
    // Allocate enhanced buffers
    voiceInputBuffer = (float*)heap_caps_malloc(ML_INPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    voiceOutputBuffer = (float*)heap_caps_malloc(ML_OUTPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    personalizedOutputBuffer = (float*)heap_caps_malloc(ML_OUTPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    
    melSpectrogramBuffer = (float*)heap_caps_malloc(MEL_BINS * VOICE_FRAME_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    voiceFeaturesBuffer = (float*)heap_caps_malloc(VOICE_FEATURES_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    pitchBuffer = (float*)heap_caps_malloc(VOICE_FRAME_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    energyBuffer = (float*)heap_caps_malloc(VOICE_FRAME_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!voiceInputBuffer || !voiceOutputBuffer || !personalizedOutputBuffer || 
        !melSpectrogramBuffer || !voiceFeaturesBuffer || !pitchBuffer || !energyBuffer) {
        logMessage("ERROR", "Failed to allocate personalization buffers");
        return;
    }
    
    // Initialize personalization config
    personalizationConfig.voiceClarity = 0.8f;
    personalizationConfig.naturalness = 0.9f;
    personalizationConfig.emotionalExpression = 0.7f;
    personalizationConfig.speechTempo = 1.0f;
    personalizationConfig.volumeAdaptation = 0.8f;
    personalizationConfig.noiseReduction = true;
    personalizationConfig.realTimeAdaptation = true;
    
    // Initialize user profiles
    for (int i = 0; i < MAX_VOICE_PROFILES; i++) {
        memset(&userProfiles[i], 0, sizeof(VoiceProfile));
        userProfiles[i].isCalibrated = false;
        userProfiles[i].speakingRate = 150.0f;
        userProfiles[i].emotionalTone = 0.5f;
    }
    
    // Load saved profiles
    loadVoiceProfiles();
    
    logMessage("PERSONALIZATION", "Personalization system initialized");
}

// ===== VOICE CALIBRATION =====
void startVoiceCalibration(int profileIndex, String userName, String userId) {
    if (profileIndex >= MAX_VOICE_PROFILES) return;
    
    logMessage("CALIBRATION", ("Starting voice calibration for profile " + String(profileIndex)).c_str());
    
    currentProfileIndex = profileIndex;
    isVoiceCalibrationMode = true;
    currentState = CALIBRATION_MODE;
    
    VoiceProfile* profile = &userProfiles[profileIndex];
    memset(profile, 0, sizeof(VoiceProfile));
    strlcpy(profile->userName, userName.c_str(), sizeof(profile->userName));
    strlcpy(profile->userId, userId.c_str(), sizeof(profile->userId));
    
    updateDisplay("Voice Calibration", "Starting...", ("User: " + userName).c_str(), "Speak naturally");
    
    // Start calibration task
    xTaskCreatePinnedToCore(
        voice_calibration_task,
        "VoiceCalibration",
        8192,
        &profileIndex,
        3,
        NULL,
        1
    );
}

void voice_calibration_task(void* parameter) {
    int profileIndex = *(int*)parameter;
    VoiceProfile* profile = &userProfiles[profileIndex];
    
    logMessage("CALIBRATION", "Voice calibration task started");
    
    float calibrationSamples[VOICE_CALIBRATION_SAMPLES][VOICE_FEATURES_SIZE];
    int sampleCount = 0;
    
    updateDisplay("Calibration", "Recording...", "Say phrases naturally", ("Sample: 0/" + String(VOICE_CALIBRATION_SAMPLES)).c_str());
    
    while (sampleCount < VOICE_CALIBRATION_SAMPLES && isVoiceCalibrationMode) {
        if (currentState == PROCESSING) {
            // Extract voice features from current audio
            extractVoiceFeatures(voiceInputBuffer, ML_INPUT_SIZE, &currentVoiceFeatures);
            
            // Convert features to flat array
            flattenVoiceFeatures(&currentVoiceFeatures, calibrationSamples[sampleCount]);
            
            sampleCount++;
            
            // Send progress via BLE
            sendVoiceCalibrationProgress(profileIndex, sampleCount, VOICE_CALIBRATION_SAMPLES);
            
            updateDisplay("Calibration", "Recording...", 
                         ("Sample: " + String(sampleCount) + "/" + String(VOICE_CALIBRATION_SAMPLES)).c_str(),
                         "Keep speaking...");
            
            logMessage("CALIBRATION", ("Collected sample " + String(sampleCount)).c_str());
            
            vTaskDelay(pdMS_TO_TICKS(500));
        }
        
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    
    bool success = false;
    if (sampleCount >= VOICE_CALIBRATION_SAMPLES) {
        updateDisplay("Calibration", "Processing...", "Analyzing voice", "Please wait...");
        
        // Analyze collected samples
        analyzeVoiceCharacteristics(calibrationSamples, sampleCount, profile);
        extractPersonalizationFeatures(calibrationSamples, sampleCount, profile);
        initializeAdaptationWeights(profile);
        
        profile->isCalibrated = true;
        profile->lastUsed = millis();
        
        // Save profile
        saveVoiceProfile(profileIndex);
        
        updateDisplay("Calibration", "Completed!", ("User: " + String(profile->userName)).c_str(), "Voice learned");
        
        logMessage("CALIBRATION", "Voice calibration completed successfully");
        success = true;
        
        // Train personalization model if available
        trainPersonalizationModel(profile);
    } else {
        updateDisplay("Calibration", "Failed", "Insufficient data", "Try again");
        logMessage("CALIBRATION", "Voice calibration failed - insufficient samples");
    }
    
    // Send completion status via BLE
    sendVoiceCalibrationComplete(profileIndex, success);
    
    isVoiceCalibrationMode = false;
    currentState = IDLE;
    
    vTaskDelay(pdMS_TO_TICKS(3000));
    vTaskDelete(NULL);
}

// ===== VOICE PROCESSING =====
bool processVoiceWithPersonalization(float* inputAudio, float* outputAudio) {
    // Copy input to voice buffer
    memcpy(voiceInputBuffer, inputAudio, ML_INPUT_SIZE * sizeof(float));
    
    // Run voice conversion
    bool conversionSuccess = runVoiceConversion(voiceInputBuffer, voiceOutputBuffer);
    
    if (conversionSuccess && isPersonalizationActive && currentProfileIndex >= 0) {
        VoiceProfile* profile = &userProfiles[currentProfileIndex];
        
        // Apply personalization
        runPersonalization(voiceOutputBuffer, profile, personalizedOutputBuffer);
        
        // Apply additional effects
        applyPersonalizationEffects(personalizedOutputBuffer, ML_OUTPUT_SIZE, profile);
        
        // Real-time adaptation
        if (personalizationConfig.realTimeAdaptation) {
            performRealTimeAdaptation(voiceInputBuffer, personalizedOutputBuffer, profile);
        }
        
        // Copy personalized output
        memcpy(outputAudio, personalizedOutputBuffer, ML_OUTPUT_SIZE * sizeof(float));
    } else {
        // Copy basic output
        memcpy(outputAudio, voiceOutputBuffer, ML_OUTPUT_SIZE * sizeof(float));
    }
    
    return conversionSuccess;
}

void extractVoiceFeatures(float* audioBuffer, int bufferSize, VoiceFeatures* features) {
    // Extract Mel-spectrogram
    computeMelSpectrogram(audioBuffer, bufferSize, features->melSpectrogram);
    
    // Extract pitch using autocorrelation
    extractPitch(audioBuffer, bufferSize, features->pitch);
    
    // Extract energy (RMS)
    extractEnergy(audioBuffer, bufferSize, features->energy);
    
    // Extract spectral features
    extractSpectralCentroid(audioBuffer, bufferSize, features->spectralCentroid);
    extractSpectralRolloff(audioBuffer, bufferSize, features->spectralRolloff);
    
    // Extract MFCC
    extractMFCC(audioBuffer, bufferSize, features->mfcc);
    
    // Extract zero crossing rate
    extractZCR(audioBuffer, bufferSize, features->zcr);
}

// ===== FEATURE EXTRACTION FUNCTIONS =====
void computeMelSpectrogram(float* audioBuffer, int bufferSize, float melSpec[][VOICE_FRAME_SIZE/VOICE_HOP_SIZE]) {
    int numFrames = bufferSize / VOICE_HOP_SIZE;
    
    for (int frame = 0; frame < numFrames; frame++) {
        int frameStart = frame * VOICE_HOP_SIZE;
        
        // Apply Hamming window
        for (int i = 0; i < VOICE_FRAME_SIZE && (frameStart + i) < bufferSize; i++) {
            float window = 0.54f - 0.46f * cos(2.0f * M_PI * i / (VOICE_FRAME_SIZE - 1));
            melSpectrogramBuffer[i] = audioBuffer[frameStart + i] * window;
        }
        
        // Convert to mel-scale
        for (int mel = 0; mel < MEL_BINS; mel++) {
            float magnitude = 0.0f;
            float melFreq = 2595.0f * log10f(1.0f + (mel * VOICE_SAMPLE_RATE / 2.0f) / (MEL_BINS * 700.0f));
            int binStart = (int)(melFreq * N_FFT / VOICE_SAMPLE_RATE);
            int binEnd = (int)((melFreq + 1) * N_FFT / VOICE_SAMPLE_RATE);
            
            for (int bin = binStart; bin < binEnd && bin < VOICE_FRAME_SIZE; bin++) {
                magnitude += melSpectrogramBuffer[bin] * melSpectrogramBuffer[bin];
            }
            
            melSpec[mel][frame] = log10f(magnitude + 1e-10f);
        }
    }
}

void extractPitch(float* audioBuffer, int bufferSize, float* pitchBuffer) {
    int maxLag = VOICE_SAMPLE_RATE / 50;   // 50 Hz minimum
    int minLag = VOICE_SAMPLE_RATE / 500;  // 500 Hz maximum
    
    for (int frame = 0; frame < bufferSize / VOICE_HOP_SIZE; frame++) {
        int frameStart = frame * VOICE_HOP_SIZE;
        float maxCorr = 0.0f;
        int bestLag = 0;
        
        for (int lag = minLag; lag < maxLag && (frameStart + lag) < bufferSize; lag++) {
            float correlation = 0.0f;
            float norm1 = 0.0f, norm2 = 0.0f;
            
            for (int i = 0; i < VOICE_FRAME_SIZE && (frameStart + i + lag) < bufferSize; i++) {
                float sample1 = audioBuffer[frameStart + i];
                float sample2 = audioBuffer[frameStart + i + lag];
                
                correlation += sample1 * sample2;
                norm1 += sample1 * sample1;
                norm2 += sample2 * sample2;
            }
            
            if (norm1 > 0 && norm2 > 0) {
                correlation /= sqrt(norm1 * norm2);
                if (correlation > maxCorr) {
                    maxCorr = correlation;
                    bestLag = lag;
                }
            }
        }
        
        pitchBuffer[frame] = bestLag > 0 ? (float)VOICE_SAMPLE_RATE / bestLag : 0.0f;
    }
}

void extractEnergy(float* audioBuffer, int bufferSize, float* energyBuffer) {
    for (int frame = 0; frame < bufferSize / VOICE_HOP_SIZE; frame++) {
        int frameStart = frame * VOICE_HOP_SIZE;
        float energy = 0.0f;
        
        for (int i = 0; i < VOICE_FRAME_SIZE && (frameStart + i) < bufferSize; i++) {
            energy += audioBuffer[frameStart + i] * audioBuffer[frameStart + i];
        }
        
        energyBuffer[frame] = sqrt(energy / VOICE_FRAME_SIZE);
    }
}

void extractSpectralCentroid(float* audioBuffer, int bufferSize, float* centroidBuffer) {
    for (int frame = 0; frame < bufferSize / VOICE_HOP_SIZE; frame++) {
        int frameStart = frame * VOICE_HOP_SIZE;
        float weightedSum = 0.0f;
        float magnitudeSum = 0.0f;
        
        for (int i = 0; i < VOICE_FRAME_SIZE && (frameStart + i) < bufferSize; i++) {
            float magnitude = abs(audioBuffer[frameStart + i]);
            float frequency = (float)i * VOICE_SAMPLE_RATE / VOICE_FRAME_SIZE;
            
            weightedSum += frequency * magnitude;
            magnitudeSum += magnitude;
        }
        
        centroidBuffer[frame] = magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0f;
    }
}

void extractSpectralRolloff(float* audioBuffer, int bufferSize, float* rolloffBuffer) {
    for (int frame = 0; frame < bufferSize / VOICE_HOP_SIZE; frame++) {
        int frameStart = frame * VOICE_HOP_SIZE;
        float totalEnergy = 0.0f;
        
        // Calculate total energy
        for (int i = 0; i < VOICE_FRAME_SIZE && (frameStart + i) < bufferSize; i++) {
            totalEnergy += audioBuffer[frameStart + i] * audioBuffer[frameStart + i];
        }
        
        // Find 85% rolloff point
        float cumulativeEnergy = 0.0f;
        float threshold = 0.85f * totalEnergy;
        
        for (int i = 0; i < VOICE_FRAME_SIZE && (frameStart + i) < bufferSize; i++) {
            cumulativeEnergy += audioBuffer[frameStart + i] * audioBuffer[frameStart + i];
            if (cumulativeEnergy >= threshold) {
                rolloffBuffer[frame] = (float)i * VOICE_SAMPLE_RATE / VOICE_FRAME_SIZE;
                break;
            }
        }
    }
}

void extractMFCC(float* audioBuffer, int bufferSize, float mfcc[][VOICE_FRAME_SIZE/VOICE_HOP_SIZE]) {
    for (int frame = 0; frame < bufferSize / VOICE_HOP_SIZE; frame++) {
        // Apply DCT to mel-spectrogram
        for (int coeff = 0; coeff < 13; coeff++) {
            float sum = 0.0f;
            for (int mel = 0; mel < MEL_BINS; mel++) {
                float melValue = currentVoiceFeatures.melSpectrogram[mel][frame];
                sum += melValue * cos(M_PI * coeff * (mel + 0.5f) / MEL_BINS);
            }
            mfcc[coeff][frame] = sum;
        }
    }
}

void extractZCR(float* audioBuffer, int bufferSize, float* zcrBuffer) {
    for (int frame = 0; frame < bufferSize / VOICE_HOP_SIZE; frame++) {
        int frameStart = frame * VOICE_HOP_SIZE;
        int zeroCrossings = 0;
        
        for (int i = 1; i < VOICE_FRAME_SIZE && (frameStart + i) < bufferSize; i++) {
            if ((audioBuffer[frameStart + i] >= 0) != (audioBuffer[frameStart + i - 1] >= 0)) {
                zeroCrossings++;
            }
        }
        
        zcrBuffer[frame] = (float)zeroCrossings / VOICE_FRAME_SIZE;
    }
}

// ===== PROFILE MANAGEMENT =====
VoiceProfile* getVoiceProfile(int index) {
    if (index >= 0 && index < MAX_VOICE_PROFILES) {
        return &userProfiles[index];
    }
    return nullptr;
}

int getCurrentProfileIndex() {
    return currentProfileIndex;
}

int getActiveProfilesCount() {
    int count = 0;
    for (int i = 0; i < MAX_VOICE_PROFILES; i++) {
        if (userProfiles[i].isCalibrated) {
            count++;
        }
    }
    return count;
}

void selectVoiceProfile(int profileIndex) {
    if (profileIndex >= 0 && profileIndex < MAX_VOICE_PROFILES && 
        userProfiles[profileIndex].isCalibrated) {
        
        currentProfileIndex = profileIndex;
        userProfiles[profileIndex].lastUsed = millis();
        isPersonalizationActive = true;
        
        logMessage("PROFILE", ("Voice profile " + String(profileIndex) + " selected").c_str());
        updateDisplay("Profile Selected", userProfiles[profileIndex].userName, 
                     ("Pitch: " + String(userProfiles[profileIndex].pitchRange[0]) + "-" + 
                      String(userProfiles[profileIndex].pitchRange[1]) + "Hz").c_str(), "Ready");
    }
}

void deleteVoiceProfile(int profileIndex) {
    if (profileIndex >= 0 && profileIndex < MAX_VOICE_PROFILES) {
        // Clear profile data
        memset(&userProfiles[profileIndex], 0, sizeof(VoiceProfile));
        userProfiles[profileIndex].isCalibrated = false;
        
        // Delete profile file
        char filename[64];
        sprintf(filename, "/voice_profile_%d.json", profileIndex);
        
        if (xSemaphoreTake(spiffsMutex, portMAX_DELAY)) {
            if (SPIFFS.exists(filename)) {
                SPIFFS.remove(filename);
                logMessage("PROFILE", ("Voice profile " + String(profileIndex) + " deleted").c_str());
            }
            xSemaphoreGive(spiffsMutex);
        }
        
        // Update current profile if needed
        if (currentProfileIndex == profileIndex) {
            currentProfileIndex = -1;
            isPersonalizationActive = false;
        }
    }
}

// ===== PERSONALIZATION EFFECTS =====
void applyPersonalizationEffects(float* audioBuffer, int bufferSize, VoiceProfile* profile) {
    if (!profile->isCalibrated) return;
    
    // Apply pitch adjustment
    float pitchRatio = profile->pitchRange[0] / 120.0f; // 120Hz default
    applyPitchShift(audioBuffer, bufferSize, pitchRatio);
    
    // Apply timbre modification
    applyTimbreAdjustment(audioBuffer, bufferSize, profile->timbreFeatures);
    
    // Apply speaking rate adjustment
    float tempoRatio = profile->speakingRate / 150.0f;  // 150 WPM default
    applyTempoAdjustment(audioBuffer, bufferSize, tempoRatio);
    
    // Apply emotional tone
    applyEmotionalTone(audioBuffer, bufferSize, profile->emotionalTone);
    
    // Apply naturalness enhancements
    if (personalizationConfig.naturalness > 0.5f) {
        applyVoiceNaturalnessEnhancements(audioBuffer, bufferSize, profile);
    }
}

void applyPitchShift(float* audioBuffer, int bufferSize, float pitchRatio) {
    if (abs(pitchRatio - 1.0f) < 0.01f) return; // Skip if no change needed
    
    // Simple pitch shifting using resampling
    int newSize = (int)(bufferSize / pitchRatio);
    float* tempBuffer = (float*)malloc(newSize * sizeof(float));
    
    if (!tempBuffer) return;
    
    for (int i = 0; i < newSize; i++) {
        float pos = i * pitchRatio;
        int idx = (int)pos;
        float frac = pos - idx;
        
        if (idx < bufferSize - 1) {
            tempBuffer[i] = audioBuffer[idx] * (1.0f - frac) + audioBuffer[idx + 1] * frac;
        } else if (idx < bufferSize) {
            tempBuffer[i] = audioBuffer[idx];
        } else {
            tempBuffer[i] = 0.0f;
        }
    }
    
    // Copy back with size adjustment
    int copySize = min(bufferSize, newSize);
    memcpy(audioBuffer, tempBuffer, copySize * sizeof(float));
    
    // Fill remaining with zeros if needed
    if (copySize < bufferSize) {
        memset(&audioBuffer[copySize], 0, (bufferSize - copySize) * sizeof(float));
    }
    
    free(tempBuffer);
}

void applyTimbreAdjustment(float* audioBuffer, int bufferSize, float* timbreFeatures) {
    // Apply spectral shaping based on timbre features
    for (int i = 0; i < bufferSize; i++) {
        float shaping = 1.0f;
        float freq = (float)i * VOICE_SAMPLE_RATE / bufferSize;
        
        if (freq < 1000) {
            shaping *= timbreFeatures[0];  // Low frequency
        } else if (freq < 3000) {
            shaping *= timbreFeatures[1];  // Mid frequency
        } else {
            shaping *= timbreFeatures[2];  // High frequency
        }
        
        audioBuffer[i] *= shaping;
    }
}

void applyTempoAdjustment(float* audioBuffer, int bufferSize, float tempoRatio) {
    if (abs(tempoRatio - 1.0f) < 0.01f) return;
    
    // Time stretching for tempo adjustment
    int newSize = (int)(bufferSize * tempoRatio);
    float* tempBuffer = (float*)malloc(newSize * sizeof(float));
    
    if (!tempBuffer) return;
    
    for (int i = 0; i < newSize; i++) {
        float pos = i / tempoRatio;
        int idx = (int)pos;
        float frac = pos - idx;
        
        if (idx < bufferSize - 1) {
            tempBuffer[i] = audioBuffer[idx] * (1.0f - frac) + audioBuffer[idx + 1] * frac;
        } else if (idx < bufferSize) {
            tempBuffer[i] = audioBuffer[idx];
        } else {
            tempBuffer[i] = 0.0f;
        }
    }
    
    int copySize = min(bufferSize, newSize);
    memcpy(audioBuffer, tempBuffer, copySize * sizeof(float));
    
    if (copySize < bufferSize) {
        memset(&audioBuffer[copySize], 0, (bufferSize - copySize) * sizeof(float));
    }
    
    free(tempBuffer);
}

void applyEmotionalTone(float* audioBuffer, int bufferSize, float emotionalTone) {
    float intensity = emotionalTone;
    
    for (int i = 0; i < bufferSize; i++) {
        // Add subtle tremolo for emotional expression
        float tremolo = 1.0f + 0.05f * intensity * sin(2.0f * M_PI * 5.0f * i / VOICE_SAMPLE_RATE);
        audioBuffer[i] *= tremolo;
        
        // Add harmonic warmth for higher emotional tones
        if (intensity > 0.5f) {
            float harmonicMix = (intensity - 0.5f) * 0.1f;
            audioBuffer[i] += harmonicMix * sin(3.0f * 2.0f * M_PI * i / VOICE_SAMPLE_RATE);
        }
    }
}

void applyVoiceNaturalnessEnhancements(float* audioBuffer, int bufferSize, VoiceProfile* profile) {
    // Add breathing sounds
    addBreathingSounds(audioBuffer, bufferSize, profile);
    
    // Add vocal tremor (natural variations)
    addVocalTremor(audioBuffer, bufferSize, profile);
    
    // Enhance formants
    enhanceFormants(audioBuffer, bufferSize, profile);
    
    // Apply prosody modeling
    applyProsodyModeling(audioBuffer, bufferSize, profile);
}

void addBreathingSounds(float* audioBuffer, int bufferSize, VoiceProfile* profile) {
    static unsigned long lastBreath = 0;
    unsigned long currentTime = millis();
    
    // Add breath sounds every 3-5 seconds
    if (currentTime - lastBreath > 3000 + (rand() % 2000)) {
        float breathIntensity = 0.02f * personalizationConfig.naturalness;
        
        for (int i = 0; i < min(bufferSize, 100); i++) {
            float breathNoise = (float(rand()) / RAND_MAX - 0.5f) * breathIntensity;
            audioBuffer[i] += breathNoise;
        }
        
        lastBreath = currentTime;
    }
}

void addVocalTremor(float* audioBuffer, int bufferSize, VoiceProfile* profile) {
    float tremorRate = 4.0f + profile->emotionalTone * 2.0f; // 4-6 Hz
    float tremorDepth = 0.03f * personalizationConfig.naturalness;
    
    for (int i = 0; i < bufferSize; i++) {
        float tremor = sin(2.0f * M_PI * tremorRate * i / VOICE_SAMPLE_RATE);
        audioBuffer[i] *= (1.0f + tremor * tremorDepth);
    }
}

void enhanceFormants(float* audioBuffer, int bufferSize, VoiceProfile* profile) {
    // Enhance vocal tract resonances
    float formantFreqs[] = {800, 1200, 2500}; // Typical formant frequencies
    
    for (int formant = 0; formant < 3; formant++) {
        float freq = formantFreqs[formant] * (0.8f + 0.4f * profile->timbreFeatures[formant]);
        enhanceFrequencyBand(audioBuffer, bufferSize, freq, 100.0f, 1.2f);
    }
}

void enhanceFrequencyBand(float* audioBuffer, int bufferSize, float centerFreq, float bandwidth, float gain) {
    // Simple bandpass enhancement
    float omega = 2.0f * M_PI * centerFreq / VOICE_SAMPLE_RATE;
    float alpha = sin(omega) * sinh(log(2.0f) / 2.0f * bandwidth * omega / sin(omega));
    
    static float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    
    float b0 = alpha * gain;
    float b1 = 0;
    float b2 = -alpha * gain;
    float a0 = 1 + alpha;
    float a1 = -2 * cos(omega);
    float a2 = 1 - alpha;
    
    for (int i = 0; i < bufferSize; i++) {
        float x0 = audioBuffer[i];
        float y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        
        audioBuffer[i] = y0;
        
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }
}

void applyProsodyModeling(float* audioBuffer, int bufferSize, VoiceProfile* profile) {
    // Apply intonation patterns and stress
    float stressPattern = sin(2.0f * M_PI * 0.5f * millis() / 1000.0f);
    float intonationGain = 1.0f + 0.1f * stressPattern * personalizationConfig.emotionalExpression;
    
    for (int i = 0; i < bufferSize; i++) {
        audioBuffer[i] *= intonationGain;
    }
}

// ===== CONFIGURATION MANAGEMENT =====
PersonalizationConfig* getPersonalizationConfig() {
    return &personalizationConfig;
}

void updatePersonalizationConfig(StaticJsonDocument<512>& doc) {
    if (doc.containsKey("voice_clarity")) {
        personalizationConfig.voiceClarity = doc["voice_clarity"];
    }
    if (doc.containsKey("naturalness")) {
        personalizationConfig.naturalness = doc["naturalness"];
    }
    if (doc.containsKey("emotional_expression")) {
        personalizationConfig.emotionalExpression = doc["emotional_expression"];
    }
    if (doc.containsKey("speech_tempo")) {
        personalizationConfig.speechTempo = doc["speech_tempo"];
    }
    if (doc.containsKey("noise_reduction")) {
        personalizationConfig.noiseReduction = doc["noise_reduction"];
    }
    if (doc.containsKey("real_time_adaptation")) {
        personalizationConfig.realTimeAdaptation = doc["real_time_adaptation"];
    }
    
    logMessage("PERSONALIZATION", "Configuration updated");
}

// ===== REAL-TIME ADAPTATION =====
void performRealTimeAdaptation(float* inputAudio, float* outputAudio, VoiceProfile* profile) {
    static unsigned long lastAdaptationTime = 0;
    unsigned long currentTime = millis();
    
    if (currentTime - lastAdaptationTime < 1000) return; // Adapt every second
    lastAdaptationTime = currentTime;
    
    // Extract current features
    VoiceFeatures currentFeatures;
    extractVoiceFeatures(inputAudio, ML_INPUT_SIZE, &currentFeatures);
    
    // Compare with user profile
    float similarity = calculateVoiceSimilarity(&currentFeatures, profile);
    
    if (similarity < VOICE_SIMILARITY_THRESHOLD) {
        // Voice has changed, adapt
        adaptVoiceProfile(&currentFeatures, profile);
        logMessage("ADAPTATION", ("Voice adapted, similarity: " + String(similarity)).c_str());
    }
}

float calculateVoiceSimilarity(VoiceFeatures* current, VoiceProfile* profile) {
    float currentFlat[VOICE_FEATURES_SIZE];
    flattenVoiceFeatures(current, currentFlat);
    
    float similarity = 0.0f;
    float norm1 = 0.0f, norm2 = 0.0f;
    
    for (int i = 0; i < PERSONALIZATION_FEATURES_SIZE; i++) {
        similarity += currentFlat[i] * profile->voiceCharacteristics[i];
        norm1 += currentFlat[i] * currentFlat[i];
        norm2 += profile->voiceCharacteristics[i] * profile->voiceCharacteristics[i];
    }
    
    if (norm1 > 0 && norm2 > 0) {
        similarity /= sqrt(norm1 * norm2);
    }
    
    return similarity;
}

void adaptVoiceProfile(VoiceFeatures* current, VoiceProfile* profile) {
    float currentFlat[VOICE_FEATURES_SIZE];
    flattenVoiceFeatures(current, currentFlat);
    
    // Update profile with exponential moving average
    for (int i = 0; i < PERSONALIZATION_FEATURES_SIZE; i++) {
        profile->voiceCharacteristics[i] = (1.0f - ADAPTATION_LEARNING_RATE) * profile->voiceCharacteristics[i] +
                                          ADAPTATION_LEARNING_RATE * currentFlat[i];
    }
    
    // Update adaptation weights
    for (int i = 0; i < 64; i++) {
        float delta = currentFlat[i % VOICE_FEATURES_SIZE] - profile->voiceCharacteristics[i % PERSONALIZATION_FEATURES_SIZE];
        profile->adaptationWeights[i] += ADAPTATION_LEARNING_RATE * delta * 0.1f;
        profile->adaptationWeights[i] = constrain(profile->adaptationWeights[i], 0.5f, 1.5f);
    }
}

// ===== UTILITY FUNCTIONS =====
void flattenVoiceFeatures(VoiceFeatures* features, float* flatArray) {
    int index = 0;
    
    // Add pitch statistics
    float pitchMean = 0, pitchStd = 0;
    calculateStatistics(features->pitch, VOICE_FRAME_SIZE/VOICE_HOP_SIZE, &pitchMean, &pitchStd);
    flatArray[index++] = pitchMean;
    flatArray[index++] = pitchStd;
    
    // Add energy statistics
    float energyMean = 0, energyStd = 0;
    calculateStatistics(features->energy, VOICE_FRAME_SIZE/VOICE_HOP_SIZE, &energyMean, &energyStd);
    flatArray[index++] = energyMean;
    flatArray[index++] = energyStd;
    
    // Add spectral centroid statistics
    float centroidMean = 0, centroidStd = 0;
    calculateStatistics(features->spectralCentroid, VOICE_FRAME_SIZE/VOICE_HOP_SIZE, &centroidMean, &centroidStd);
    flatArray[index++] = centroidMean;
    flatArray[index++] = centroidStd;
    
    // Add MFCC coefficients
    for (int i = 0; i < 13 && index < VOICE_FEATURES_SIZE; i++) {
        float mfccMean = 0;
        for (int j = 0; j < VOICE_FRAME_SIZE/VOICE_HOP_SIZE; j++) {
            mfccMean += features->mfcc[i][j];
        }
        flatArray[index++] = mfccMean / (VOICE_FRAME_SIZE/VOICE_HOP_SIZE);
    }
    
    // Add zero crossing rate
    float zcrMean = 0;
    for (int i = 0; i < VOICE_FRAME_SIZE/VOICE_HOP_SIZE; i++) {
        zcrMean += features->zcr[i];
    }
    flatArray[index++] = zcrMean / (VOICE_FRAME_SIZE/VOICE_HOP_SIZE);
    
    // Fill remaining with zeros
    while (index < VOICE_FEATURES_SIZE) {
        flatArray[index++] = 0.0f;
    }
}

void calculateStatistics(float* data, int size, float* mean, float* std) {
    *mean = 0.0f;
    for (int i = 0; i < size; i++) {
        *mean += data[i];
    }
    *mean /= size;
    
    *std = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - *mean;
        *std += diff * diff;
    }
    *std = sqrt(*std / size);
}

// Placeholder functions for voice analysis
void analyzeVoiceCharacteristics(float calibrationSamples[][VOICE_FEATURES_SIZE], int sampleCount, VoiceProfile* profile) {
    // Analyze voice characteristics from calibration samples
    // Implementation would include advanced voice analysis
    logMessage("ANALYSIS", "Analyzing voice characteristics");
}

void extractPersonalizationFeatures(float calibrationSamples[][VOICE_FEATURES_SIZE], int sampleCount, VoiceProfile* profile) {
    // Extract personalization features
    logMessage("ANALYSIS", "Extracting personalization features");
}

void initializeAdaptationWeights(VoiceProfile* profile) {
    // Initialize adaptation weights
    for (int i = 0; i < 64; i++) {
        profile->adaptationWeights[i] = 1.0f;
    }
}

void loadVoiceProfiles() {
    // Load voice profiles from SPIFFS
    logMessage("PROFILE", "Loading voice profiles");
}

void saveVoiceProfile(int profileIndex) {
    // Save voice profile to SPIFFS
    logMessage("PROFILE", ("Saving voice profile " + String(profileIndex)).c_str());
}

void trainPersonalizationModel(VoiceProfile* profile) {
    // Train personalization model
    logMessage("ML", "Training personalization model");
}