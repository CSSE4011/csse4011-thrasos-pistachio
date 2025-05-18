// Receive instructions from classification.c
// Move servo depending on classification

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>

uint8_t servo_pos; // Global variable for position of servo (0, 1 or 2)

// A14/ P1.15 = Pan
// A16/ P1.13 = Tilt
#define PAN_PIN 15
#define TILT_PIN 13

#define GPIO_A_PORT DT_NODELABEL(gpio1)

const struct device *gpioa_dev = DEVICE_DT_GET(GPIO_A_PORT);

// Use PWM to control the servo https://docs.zephyrproject.org/latest/samples/basic/servo_motor/README.html#servo-motor

void command_receive_thread(void) {

}

void test(void) {
    while(1) {
        gpio_pin_set(gpioa_dev, PAN_PIN, 0);
        k_msleep(50);
        gpio_pin_set(gpioa_dev, PAN_PIN, 0);
        k_busy_wait(50);
    }
}

void initialise_pins(void) {
    if (!device_is_ready(gpioa_dev)) {
        printk("GPIO device not ready\n");
        return;
    }

    gpio_pin_configure(gpioa_dev, PAN_PIN, GPIO_OUTPUT_INACTIVE | GPIO_ACTIVE_HIGH);
    gpio_pin_configure(gpioa_dev, TILT_PIN, GPIO_OUTPUT_INACTIVE | GPIO_ACTIVE_HIGH);
}