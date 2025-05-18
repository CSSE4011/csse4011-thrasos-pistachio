// Receive instructions from classification.c
// Move servo depending on classification

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/pwm.h>

uint8_t servo_pos; // Global variable for position of servo (0, 1 or 2)

// A14/ P0.13 = Pan
// A16/ P0.14 = Tilt
#define PAN_PIN 13
#define TILT_PIN 14

#define PWM_CTL_NODE DT_NODELABEL(pwm0)

const struct device *pwm_dev = DEVICE_DT_GET(PWM_CTL_NODE);

// Use PWM to control the servo https://blog.wokwi.com/learn-servo-motor-using-wokwi-logic-analyzer/
#define PWM_PERIOD_USEC 20000  // 50Hz period = 20ms (20000us)

void set_angle(uint8_t pin, uint8_t angle) {
    uint32_t pulse_width = 1000 + ((uint32_t)angle * 1000) / 180; // Pulse width will be between 1000-2000 us (1ms = 0 degrees, 2ms = 180 degrees)
    pwm_set(pwm_dev, pin, PWM_PERIOD_USEC, pulse_width, 0); // set pwm signal
}

void command_receive_thread(void) {

}

void test(void) {
    while (1) {
        for (int angle = 0; angle <= 180; angle += 10) {
            set_angle(PAN_PIN, angle);
            set_angle(TILT_PIN, 180 - angle);
            k_msleep(500);
        }

        for (int angle = 180; angle >= 0; angle -= 10) {
            set_angle(PAN_PIN, angle);
            set_angle(TILT_PIN, 180 - angle);
            k_msleep(500);
        }
    }
}

void initialise_pwm(void) {
    if (!device_is_ready(pwm_dev)) {
        printk("PWM device not ready\n");
    } else {
        printk("PWM ready\n");
    }
}