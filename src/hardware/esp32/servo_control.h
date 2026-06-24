#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#include <Arduino.h>

/**
 * @file servo_control.h
 * @brief Positional servo control for ESP32 using LEDC hardware PWM.
 *
 * Moves the servo to a target angle (0–180°) on command.
 * Replaces the earlier continuous-sweep implementation.
 *
 * SG90 timing at 50 Hz with 16-bit LEDC resolution:
 *   - 0°   → 544 µs pulse  → duty = (544 / 20000) * 65536 ≈ 1782
 *   - 180° → 2400 µs pulse → duty = (2400 / 20000) * 65536 ≈ 7864
 */

// SG90 servo timing constants (ESP32 LEDC, 16-bit, 50 Hz)
constexpr uint32_t SERVO_FREQ      = 50;    // 50 Hz PWM
constexpr uint8_t  LEDC_RESOLUTION = 16;    // 16-bit

// Pulse widths for SG90
constexpr uint32_t PULSE_0_DEG    = 544;   // µs
constexpr uint32_t PULSE_180_DEG  = 2400;  // µs
constexpr uint32_t PULSE_WIDTH_US = 20000; // 20 ms period (50 Hz)

// Derived duty values (duty = pulse_us / 20000 * 65536)
constexpr uint32_t DUTY_0_DEG   = (PULSE_0_DEG   * 65536) / PULSE_WIDTH_US;
constexpr uint32_t DUTY_180_DEG = (PULSE_180_DEG * 65536) / PULSE_WIDTH_US;
constexpr uint32_t DUTY_RANGE   = DUTY_180_DEG - DUTY_0_DEG;
constexpr int      MAX_ANGLE    = 180;

/**
 * @brief Positional servo controller using LEDC hardware PWM.
 *
 * Usage:
 * @code
 *   ServoControl servo;
 *   servo.begin(GPIO_NUM_13);
 *   servo.setAngle(90);   // centre
 *   servo.setAngle(180);  // far end
 * @endcode
 */
class ServoControl {
public:
  ServoControl() : _pin(-1), _currentAngle(-1) {}

  /**
   * @brief Attach LEDC PWM to the given pin.
   * @param pin  GPIO pin for servo signal wire.
   */
  void begin(int pin) {
    _pin = pin;
    ledcAttach(pin, SERVO_FREQ, LEDC_RESOLUTION);
    setAngle(90);  // start at centre
  }

  /**
   * @brief Move the servo to a specific angle.
   * @param angle Target angle in degrees, clamped to [0, 180].
   */
  void setAngle(int angle) {
    angle = constrain(angle, 0, MAX_ANGLE);
    uint32_t duty = DUTY_0_DEG + (DUTY_RANGE * angle) / MAX_ANGLE;
    ledcWrite(_pin, duty);
    _currentAngle = angle;
  }

  /** @brief Return the last-set servo angle, or -1 if not configured. */
  int getCurrentAngle() const { return _currentAngle; }

  /** @brief Detach servo PWM from its pin. */
  void detach() { ledcDetach(_pin); }

private:
  int  _pin;
  int  _currentAngle;
};

#endif  // SERVO_CONTROL_H
