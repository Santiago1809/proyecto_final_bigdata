#ifndef BUZZER_CONTROL_H
#define BUZZER_CONTROL_H

#include <Arduino.h>

/**
 * @file buzzer_control.h
 * @brief Active buzzer control for bottle-detection feedback.
 *
 * Produces a short beep when a bottle is detected.
 * Uses non-blocking timing (millis()) so the main loop is not
 * blocked during the beep duration.
 */

class BuzzerControl {
public:
  /**
   * @brief Construct buzzer control on the given GPIO pin.
   * @param pin  GPIO pin for the active buzzer (HIGH = on).
   */
  BuzzerControl(uint8_t pin) : _pin(pin), _beepEndMs(0) {}

  /**
   * @brief Configure the buzzer pin as an output and ensure it's OFF.
   */
  void begin() {
    pinMode(_pin, OUTPUT);
    digitalWrite(_pin, LOW);
  }

  /**
   * @brief Trigger a non-blocking beep for the given duration.
   * @param durationMs  Beep length in milliseconds (default 200).
   */
  void beep(unsigned long durationMs = 200) {
    digitalWrite(_pin, HIGH);
    _beepEndMs = millis() + durationMs;
  }

  /**
   * @brief Call this from loop() to turn the buzzer OFF after
   *        the beep duration has elapsed.
   */
  void update() {
    if (_beepEndMs > 0 && millis() >= _beepEndMs) {
      digitalWrite(_pin, LOW);
      _beepEndMs = 0;
    }
  }

private:
  uint8_t       _pin;
  unsigned long _beepEndMs;  // 0 = not beeping
};

#endif  // BUZZER_CONTROL_H
