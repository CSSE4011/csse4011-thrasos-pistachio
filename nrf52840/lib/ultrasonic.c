// Take ultrasonic measurements
// Send to display data

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>
#include <zephyr/posix/sys/time.h>
#include <sys/_timeval.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/types.h>
#include <stddef.h>
#include <zephyr/sys/util.h>

#define TRIG_PIN 16
#define ECHO_PIN 15
#define GPIO0_NODE DT_NODELABEL(gpio0)

#define BIN_DEPTH_CM 50

#define STACKSIZE 1024
#define SAMPLE_FILL_LEVEL_PRIORITY 8

const struct device *gpio0 = DEVICE_DT_GET(GPIO0_NODE);

K_MSGQ_DEFINE(fill_level_msgq, sizeof(float), 10, 4);

void sample_fill_level_thread(void);
K_THREAD_DEFINE(sample_fill_level_tid, STACKSIZE, sample_fill_level_thread, NULL, NULL, NULL, SAMPLE_FILL_LEVEL_PRIORITY, 0, 0);

void ultrasonic_setup(void) {
    if (!device_is_ready(gpio0)) {
        printk("GPIO0 not ready\n");
        return;
    }

    gpio_pin_configure(gpio0, TRIG_PIN, GPIO_OUTPUT_INACTIVE | GPIO_ACTIVE_HIGH);
    gpio_pin_set(gpio0, TRIG_PIN, 0);  // Ensure low initially

    gpio_pin_configure(gpio0, ECHO_PIN, GPIO_INPUT);
}

void trigger_pulse(void) {
    gpio_pin_set(gpio0, TRIG_PIN, 0);
    k_busy_wait(2);
    gpio_pin_set(gpio0, TRIG_PIN, 1);
    k_busy_wait(10);
    gpio_pin_set(gpio0, TRIG_PIN, 0);
}

uint32_t measure_echo_duration_us(void) {
    // Wait for echo pin to go high
    while (gpio_pin_get(gpio0, ECHO_PIN) == 0);
    uint32_t start = k_cycle_get_32();

    // Wait for echo pin to go low
    while (gpio_pin_get(gpio0, ECHO_PIN) == 1);
    uint32_t end = k_cycle_get_32();

    uint32_t cycles = end - start;
    return (uint32_t)k_cyc_to_us_floor64(cycles);
}

void sample_fill_level_thread(void) {
    float fill_level;

    while(1) {
        trigger_pulse();
        uint32_t duration_us = measure_echo_duration_us();
        float distance_cm = (duration_us * 0.0343) / 2.0;
        fill_level = distance_cm / BIN_DEPTH_CM;
    
        k_msgq_put(&fill_level_msgq, &fill_level, K_NO_WAIT);

        k_msleep(100);
    }
}