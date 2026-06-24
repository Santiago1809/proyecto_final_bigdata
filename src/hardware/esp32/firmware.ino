/**
 * @file main.cpp
 * @brief ESP32 firmware for bottle detection servo scan.
 *
 * Receives JSON commands over USB serial (9600 baud), drives green/red
 * LEDs based on classification, and runs a continuous 180° servo sweep
 * via hardware timer.
 *
 * Command format:
 *   {"b":1}\n  → Green LED (bottle detected)
 *   {"b":0}\n  → Red LED (no bottle detected)
 *
 * Autonomous fallback:
 *   If no serial command is received for >= 5 seconds, the servo
 *   continues sweeping and both LEDs turn off until a new command
 *   arrives.
 *
 * Wiring:
 *   - Green LED → GPIO 26 (via 220Ω resistor)
 *   - Red LED   → GPIO 27 (via 220Ω resistor)
 *   - Servo     → GPIO 13 (signal wire)
 */

#include "led_control.h"
#include "servo_sweep.h"

// Pin assignments
constexpr uint8_t PIN_GREEN_LED = 26;
constexpr uint8_t PIN_RED_LED   = 27;
constexpr uint8_t PIN_SERVO     = 13;

// Serial
constexpr unsigned long SERIAL_BAUD = 9600;

// Global objects
LEDControl   leds(PIN_GREEN_LED, PIN_RED_LED);
ServoSweep   sweep;

// Buffer for serial input
constexpr uint8_t SERIAL_BUF_SIZE = 32;
static char       serialBuffer[SERIAL_BUF_SIZE];
static uint8_t    serialIndex = 0;

void processCommand(const char* json);

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(50);

  leds.begin();
  sweep.begin(PIN_SERVO);
  sweep.start();

  // Signal ready with a brief blink
  leds.off();
  delay(200);
  leds.off();
}

void loop() {
  // --- Process incoming serial commands ---
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n') {
      serialBuffer[serialIndex] = '\0';  // null-terminate

      if (serialIndex > 0) {
        processCommand(serialBuffer);
      }

      serialIndex = 0;  // reset buffer
    } else if (serialIndex < SERIAL_BUF_SIZE - 1) {
      serialBuffer[serialIndex++] = c;
    }
    // Silently discard if buffer overflows
  }

  // --- Standby / timeout handling ---
  if (leds.isTimedOut()) {
    // Autonomous fallback: sweep continues, LEDs go dark
    leds.off();
  } else {
    // Normal operation: standby blink pattern if idle
    leds.standbyBlink();
  }

  // Small delay to prevent tight-loop watchdog issues
  delay(5);
}

/**
 * @brief Parse a JSON command string and update LED state.
 *
 * Expected format: {"b":0} or {"b":1}
 *
 * @param json Null-terminated C string containing the JSON payload.
 */
void processCommand(const char* json) {
  // Simple JSON parser — looks for key "b" and integer value 0 or 1.
  // Using ArduinoJson would be cleaner but we keep dependencies minimal.

  const char* key = strstr(json, "\"b\"");
  if (key == nullptr) return;

  const char* colon = strchr(key, ':');
  if (colon == nullptr) return;

  // Skip whitespace and read the value
  const char* value = colon + 1;
  while (*value == ' ' || *value == '\t') ++value;

  int bottleFlag = (*value == '1') ? 1 : 0;
  leds.refresh();

  if (bottleFlag) {
    leds.green();
  } else {
    leds.red();
  }
}
