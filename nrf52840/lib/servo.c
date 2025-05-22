// Receive instructions from classification.c
// Move servo depending on classification

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/pwm.h>

uint8_t servo_pos; // Global variable for position of servo (0, 1 or 2)

#define PAN_CHANNEL 0

static const struct device *pwm_dev = DEVICE_DT_GET(DT_NODELABEL(pwm0));

#define PWM_PERIOD_NSEC 20000000  // 20ms = 50Hz

void set_position(uint8_t pos) {
    uint32_t angle = pos * 90;
    uint32_t pulse_width_usec = 1000 + (angle * 1000) / 180; // 1ms to 2ms
    uint32_t pulse_width_nsec = pulse_width_usec * 1000;

    pwm_set(pwm_dev, PAN_CHANNEL, PWM_PERIOD_NSEC, pulse_width_nsec, 0);
}

void initialise_servo(void) {

}

void test(void) {
    while (1) {
        for (int pos = 0; pos <= 2; pos += 1) {
            printk("setting angle\n");
            set_position(pos);
            k_msleep(1000);
        }
    }
}