#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/drivers/led.h>
#include "thingy.h"
 
 
#define LED0_NODE DT_ALIAS(led0)
// static const struct led_dt_spec led_0 = LED_DT_SPEC_GET(LED0_NODE);

#define LED1_NODE DT_ALIAS(led1)
// static const struct led_dt_spec led_1 = LED_DT_SPEC_GET(LED1_NODE);

#define LED2_NODE DT_ALIAS(led2)
// static const struct led_dt_spec led_2 = LED_DT_SPEC_GET(LED2_NODE);

static const struct gpio_dt_spec led_0 = GPIO_DT_SPEC_GET(LED0_NODE, gpios);
static const struct gpio_dt_spec led_1 = GPIO_DT_SPEC_GET(LED1_NODE, gpios);
static const struct gpio_dt_spec led_2 = GPIO_DT_SPEC_GET(LED2_NODE, gpios);

int colours[6][3] = {{0,0,1}, {0,1,1}, {0,1,0}, {1,1,0}, {1,0,1}, {1,0,0}};
//blue, cyan, green, yellow, magenta, red

void set_colour (int* colour) {
    //R-0, G-1, B-2
    gpio_pin_set_dt(&led_0, colour[0]);
    gpio_pin_set_dt(&led_1, colour[1]);
    gpio_pin_set_dt(&led_2, colour[2]);
}

void led_init (void) {
    gpio_pin_configure_dt(&led_0, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&led_1, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&led_2, GPIO_OUTPUT_ACTIVE);
}

#define CO2_MIN 400
#define CO2_MAX 8192
#define TVOC_MIN 0
#define TVOC_MAX 1187

void gas_colour_thread(void *p1, void *p2, void *p3) {
    // int data[2];
    int range = (CO2_MAX - CO2_MIN) / 6;
    int selected_colour = 0;
    sensor_data_t received_data;

    while (1) {
        k_mutex_lock(&sensor_mutex, K_FOREVER);
        received_data = get_sensor_data(GAS_DID);
        k_mutex_unlock(&sensor_mutex);

        int value = received_data.sensor.val1;

        selected_colour = (value - CO2_MIN) / range;

        if (selected_colour < 0) {
            selected_colour = 0;
        } else if (selected_colour > 5) {
            selected_colour = 5;
        }

        // Set colour
        set_colour(colours[selected_colour]);

        k_sleep(K_SECONDS(1));
    }
}

K_THREAD_DEFINE(gas_colour, 1024, gas_colour_thread, NULL, NULL, NULL, 7, 0, 0);
 