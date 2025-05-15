#ifndef SENSOR_SAMPLING_H
#define SENSOR_SAMPLING_H

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/util.h>
#include <zephyr/sys/printk.h>
#include <zephyr/data/json.h>
#include <zephyr/shell/shell.h>
#include "thingy.h"

#define BUTTON_NODE DT_ALIAS(sw0)
extern const struct gpio_dt_spec button;
extern struct gpio_callback button_cb_data;

#define SAMPLING_STACK_SIZE 2048
extern K_THREAD_STACK_DEFINE(sampling_stack, SAMPLING_STACK_SIZE);
extern struct k_thread sampling_thread;

struct sensor_info {
	int did;
	char *time;
	int data[2];
    char *data_str;
	size_t data_len; 
};

extern const struct json_obj_descr sensor_descr[3];

void sample_sensor(void *p1, void *p2, void *p3);

void button_init(void);
int cmd_sample(const struct shell *shell, size_t argc, char **argv);

#endif 