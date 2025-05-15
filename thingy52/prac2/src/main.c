
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/rtc.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/counter.h>
#include <zephyr/sys/util.h>
#include <zephyr/shell/shell.h>
#include <SEGGER_RTT.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/sensor/ccs811.h>
#include <stdio.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/shell/shell_uart.h>
#include <zephyr/sys/printk.h>
#include <zephyr/logging/log.h>
#include <zephyr/data/json.h>
#include <zephyr/drivers/led.h>
#include "rtc.h"
#include "thingy.h"
#include "button.h"
#include "led.h"


int cmd_sensor(const struct shell *shell, size_t argc, char **argv) {
	int did = atoi(argv[1]);
	sensor_data_t received_data;

	k_mutex_lock(&sensor_mutex, K_FOREVER);
    received_data = get_sensor_data(did);
    k_mutex_unlock(&sensor_mutex);

	if (did == 3) {
		printk ("DID: %d, data: %d, %d\n", did, received_data.sensor.val1, received_data.sensor.val2);
	} else {
		printk ("DID: %d, data: %d.%2d\n", did, received_data.sensor.val1, received_data.sensor.val2);
	}
	
	return 0;
}

int main(void) {

	rtc_init();

	date_time_t initial_time = {
        .year = 2025,
        .month = 4,
        .day = 3,
        .hour = 11,
        .minute = 30,
        .second = 0
    };
    set_date_time(&initial_time);
	
	SEGGER_RTT_Init();

	printk("started!");

	k_mutex_init(&sensor_mutex);
	button_init();
	led_init();
	set_colour(0);
	
	SHELL_CMD_ARG_REGISTER(rtc, NULL, "rtc", cmd_rtc, 2, 4);

    SHELL_CMD_ARG_REGISTER(sensor, NULL, "sensor info", cmd_sensor, 2, 2);

	SHELL_CMD_ARG_REGISTER(sample, NULL, "sample", cmd_sample, 2, 3);

    return 0;
}