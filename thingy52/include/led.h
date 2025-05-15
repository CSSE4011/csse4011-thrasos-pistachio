#ifndef LED_H
#define LED_H

#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/drivers/led.h>
#include "thingy.h"
 
 
void set_colour (uint32_t colour);

void led_init (void);

#endif