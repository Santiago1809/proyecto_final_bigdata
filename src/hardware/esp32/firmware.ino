/**
 * @file firmware.ino
 * @brief ESP32 firmware for bottle detection with positional servo.
 *
 * Receives JSON commands over USB serial (9600 baud). Drives green/red
 * LEDs based on classification, and positions the servo to the angle
 * specified in the command.
 *
 * Command format:
 *   {"b":1,"s":90}\n   → Green LED, servo at 90° (bottle detected)
 *   {"b":0,"s":180}\n  → Red LED, servo at 180° (no bottle)
 *
 * Autonomous fallback:
 *   If no serial command is received for >= 5 seconds, the servo
 *   stays at its last position and both LEDs turn off until a new
 *   command arrives.
 *
 * Wiring:
 *   - Green LED → GPIO 26 (via 220Ω resistor)
 *   - Red LED   → GPIO 27 (via 220Ω resistor)
 *   - Servo     → GPIO 13 (signal wire)
 */

#include "led_control.h"
#include "servo_control.h"

// Pin assignments
constexpr uint8_t PIN_GREEN_LED = 26;
constexpr uint8_t PIN_RED_LED   = 27;
constexpr uint8_t PIN_SERVO     = 13;

// Serial
constexpr unsigned long SERIAL_BAUD = 9600;

// Global objects
LEDControl     leds(PIN_GREEN_LED, PIN_RED_LED);
ServoControl   servo;

// Buffer for serial input
constexpr uint8_t SERIAL_BUF_SIZE = 32;
static char       serialBuffer[SERIAL_BUF_SIZE];
static uint8_t    serialIndex = 0;

void processCommand(const char* json);

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(50);

  leds.begin();
  servo.begin(PIN_SERVO);

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
    // Autonomous fallback: LEDs go dark, servo stays at last position
    leds.off();
  } else {
    // Normal operation: standby blink pattern if idle
    leds.standbyBlink();
  }

  // Small delay to prevent tight-loop watchdog issues
  delay(5);
}

/**
 * @brief Parse a JSON command string, update LED and servo.
 *
 * Expected format: {"b":0|1,"s":<angle>}
 *
 * @param json Null-terminated C string containing the JSON payload.
 */
void processCommand(const char* json) {
  // --- Parse "b" key (bottle flag) ---
  const char* bKey = strstr(json, "\"b\"");
  if (bKey == nullptr) return;

  const char* colon = strchr(bKey, ':');
  if (colon == nullptr) return;

  const char* value = colon + 1;
  while (*value == ' ' || *value == '\t') ++value;

  int bottleFlag = (*value == '1') ? 1 : 0;
  leds.refresh();

  if (bottleFlag) {
    leds.green();
  } else {
    leds.red();
  }

  // --- Parse "s" key (servo angle) ---
  const char* sKey = strstr(json, "\"s\"");
  if (sKey != nullptr) {
    const char* sColon = strchr(sKey, ':');
    if (sColon != nullptr) {
      const char* sVal = sColon + 1;
      while (*sVal == ' ' || *sVal == '\t') ++sVal;
      int angle = atoi(sVal);
      if (angle >= 0 && angle <= 180) {
        servo.setAngle(angle);
      }
    }
  }
}
