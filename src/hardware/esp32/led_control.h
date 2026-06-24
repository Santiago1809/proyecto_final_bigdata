#ifndef LED_CONTROL_H
#define LED_CONTROL_H

#include <Arduino.h>

/**
 * @file led_control.h
 * @brief Green/Red LED GPIO helpers for bottle detection signaling.
 *
 * Pin assignments (configurable via constructor):
 *   - Green LED: indicates a bottle was detected.
 *   - Red LED:   indicates no bottle detected.
 *
 * Standby mode blinks both LEDs briefly every 5 seconds when no
 * serial command has been received for >= 5 seconds.
 */

class LEDControl {
public:
  /**
   * @brief Construct LED control on the given GPIO pins.
   * @param greenPin GPIO for green LED (active HIGH).
   * @param redPin   GPIO for red LED (active HIGH).
   */
  LEDControl(uint8_t greenPin, uint8_t redPin)
      : _greenPin(greenPin), _redPin(redPin) {}

  /**
   * @brief Configure GPIO pins as outputs and turn both LEDs off.
   */
  void begin() {
    pinMode(_greenPin, OUTPUT);
    pinMode(_redPin, OUTPUT);
    off();
  }

  /** @brief Turn green LED on, red LED off. */
  void green() {
    digitalWrite(_greenPin, HIGH);
    digitalWrite(_redPin, LOW);
    _lastCommandMs = millis();
  }

  /** @brief Turn red LED on, green LED off. */
  void red() {
    digitalWrite(_redPin, HIGH);
    digitalWrite(_greenPin, LOW);
    _lastCommandMs = millis();
  }

  /** @brief Turn both LEDs off. */
  void off() {
    digitalWrite(_greenPin, LOW);
    digitalWrite(_redPin, LOW);
  }

  /**
   * @brief Perform standby blink pattern if no command received for >= 5s.
   *
   * Call this from loop(). It blinks both LEDs ON for 100ms then OFF,
   * rate-limited to once every 5 seconds.
   */
  void standbyBlink() {
    unsigned long now = millis();
    if (now - _lastCommandMs < STANDBY_TIMEOUT_MS) {
      return;  // command received recently — normal operation
    }
    if (now - _lastBlinkMs < BLINK_INTERVAL_MS) {
      return;  // rate-limited
    }
    _lastBlinkMs = now;
    digitalWrite(_greenPin, HIGH);
    digitalWrite(_redPin, HIGH);
    delay(100);
    digitalWrite(_greenPin, LOW);
    digitalWrite(_redPin, LOW);
  }

  /**
   * @brief Check whether serial timeout has been exceeded.
   * @return true if no command received for >= STANDBY_TIMEOUT_MS.
   */
  bool isTimedOut() const {
    return (millis() - _lastCommandMs) >= STANDBY_TIMEOUT_MS;
  }

  /** @brief Reset the last-command timestamp (call when a command arrives). */
  void refresh() { _lastCommandMs = millis(); }

private:
  static const unsigned long STANDBY_TIMEOUT_MS = 5000;
  static const unsigned long BLINK_INTERVAL_MS   = 5000;

  uint8_t _greenPin;
  uint8_t _redPin;
  unsigned long _lastCommandMs = 0;
  unsigned long _lastBlinkMs   = 0;
};

#endif  // LED_CONTROL_H
