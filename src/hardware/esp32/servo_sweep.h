#ifndef SERVO_SWEEP_H
#define SERVO_SWEEP_H

#include <Arduino.h>

/**
 * @file servo_sweep.h
 * @brief Continuous 180-degree servo sweep using ESP32 LEDC hardware PWM.
 *
 * Uses LEDC channel 0 at 50 Hz with 16-bit resolution. The duty cycle
 * maps to servo angle:
 *   - 0°   → 544 µs pulse  → duty = (544 / 20000) * 65536 ≈ 1782
 *   - 180° → 2400 µs pulse → duty = (2400 / 20000) * 65536 ≈ 7864
 *
 * A hardware timer advances the angle by 1° every ~167 ms, completing
 * a full 180° sweep (0→180→0) in approximately 30 seconds.
 *
 * The sweep runs autonomously once started — no loop() calls needed.
 */

// SG90 servo timing constants (ESP32 LEDC, 16-bit, 50 Hz)
constexpr uint32_t SERVO_FREQ         = 50;       // 50 Hz PWM
constexpr uint8_t  LEDC_CHANNEL       = 0;
constexpr uint8_t  LEDC_RESOLUTION    = 16;       // 16-bit
constexpr uint32_t LEDC_MAX_DUTY      = 65535;
constexpr uint32_t PULSE_WIDTH_US     = 20000;    // 20 ms period (50 Hz)

// Pulse widths for SG90
constexpr uint32_t PULSE_0_DEG    = 544;   // µs
constexpr uint32_t PULSE_180_DEG  = 2400;  // µs

// Derived duty values (duty = pulse_us / 20000 * 65536)
constexpr uint32_t DUTY_0_DEG   = (PULSE_0_DEG   * (LEDC_MAX_DUTY + 1)) / PULSE_WIDTH_US;
constexpr uint32_t DUTY_180_DEG = (PULSE_180_DEG * (LEDC_MAX_DUTY + 1)) / PULSE_WIDTH_US;
constexpr uint32_t DUTY_RANGE   = DUTY_180_DEG - DUTY_0_DEG;

// Timing
constexpr float    MS_PER_DEGREE  = 166.67f;  // ~167 ms per degree for 30s / 180°
constexpr int      SWEEP_DEGREES  = 180;

/**
 * @brief Servo sweep controller using ESP32 LEDC + hardware timer.
 *
 * The timer ISR increments the angle by 1° every ~167 ms.
 * The sweep direction reverses at 0° and 180°.
 */
class ServoSweep {
public:
  ServoSweep() : _pin(-1), _angle(0), _direction(1), _timer(nullptr) {}

  /**
   * @brief Configure LEDC PWM on the given pin and channel.
   * @param pin     GPIO pin for servo signal wire.
   * @param channel LEDC channel (0-7). Defaults to LEDC_CHANNEL.
   */
  void begin(int pin, uint8_t channel = LEDC_CHANNEL) {
    _pin = pin;
    _channel = channel;
    ledcSetup(_channel, SERVO_FREQ, LEDC_RESOLUTION);
    ledcAttachPin(_pin, _channel);
    setAngle(0);
  }

  /**
   * @brief Start the hardware timer to drive the sweep.
   *
   * The timer fires every ~167 ms and advances the angle by 1°.
   */
  void start() {
    if (_timer != nullptr) return;  // already started

    _timer = timerBegin(0, 80, true);  // prescaler 80 → 1 µs per tick
    timerAttachInterrupt(_timer, &onTimer, true);
    timerAlarmWrite(_timer, static_cast<uint64_t>(MS_PER_DEGREE * 1000), true);
    timerAlarmEnable(_timer);
  }

  /** @brief Stop the sweep timer and detach servo PWM. */
  void stop() {
    if (_timer != nullptr) {
      timerAlarmDisable(_timer);
      timerDetachInterrupt(_timer);
      timerEnd(_timer);
      _timer = nullptr;
    }
    ledcDetachPin(_pin);
  }

  /** @brief Return current servo angle (0–180). */
  int getCurrentAngle() const { return _angle; }

  /** @brief Reset sweep to 0° and set direction to forward. */
  void reset() {
    _direction = 1;
    _angle = 0;
    setAngle(0);
  }

private:
  /**
   * @brief ISR callback — advances angle by 1° every tick.
   *
   * Marked IRAM_ATTR for execution from IRAM.
   * Reverses direction at 0° and 180°.
   */
  static void IRAM_ATTR onTimer() {
    // Advance angle
    _angle += _direction;

    // Reverse direction at limits
    if (_angle >= SWEEP_DEGREES) {
      _angle = SWEEP_DEGREES - 1;
      _direction = -1;
    } else if (_angle <= 0) {
      _angle = 0;
      _direction = 1;
    }

    setAngle(_angle);
  }

  /**
   * @brief Set the servo to a specific angle via LEDC duty.
   * @param angle Target angle (0–180), clamped internally.
   */
  static void setAngle(int angle) {
    angle = constrain(angle, 0, SWEEP_DEGREES);
    uint32_t duty = DUTY_0_DEG + (DUTY_RANGE * angle) / SWEEP_DEGREES;
    ledcWrite(LEDC_CHANNEL, duty);
  }

  int          _pin;
  uint8_t      _channel;
  volatile static int   _angle;
  volatile static int   _direction;
  hw_timer_t * _timer;
};

// Static volatile members
volatile int ServoSweep::_angle     = 0;
volatile int ServoSweep::_direction = 1;

#endif  // SERVO_SWEEP_H
