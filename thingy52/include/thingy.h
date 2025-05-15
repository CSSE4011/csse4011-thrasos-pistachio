#ifndef THINGY_H
#define THINGY_H

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/sensor/ccs811.h>
// #include <drv_gas_sensor.h>drv_humidity.hdrv_pressure.h
// #include <zephyr/drivers/sensor/ccs811.h>
// #include <zephyr/drivers/sensor/hts221.h>
// #include <zephyr/drivers/sensor/lps33hb_press.h>
#include <stdio.h>
#include <zephyr/sys/ring_buffer.h>

#define BUFFER_SIZE 20
// #define CSS_SIZE sizeof(struct ccs811_result_type)
#define SENSOR_SIZE sizeof(struct sensor_value)

typedef struct {
    struct sensor_value sensor;
    // struct sensor_value sensor2;
    char timestamp[32];
} sensor_data_t;

#define TEMP_DID 0
#define HUM_DID 1
#define PRES_DID 2
#define GAS_DID 3
#define ALL_DID 15

extern struct ring_buf gas_ring;
extern struct ring_buf temp_ring;
extern struct ring_buf hum_ring;
extern struct ring_buf pres_ring;

extern struct k_mutex sensor_mutex;

void gas_read_thread(void *p1, void *p2, void *p3);

void temp_hum_read_thread(void *p1, void *p2, void *p3);

void pressure_read_thread(void *p1, void *p2, void *p3);

// void get_sensor_data (int did, int* result, char* time_str, size_t buf_size);
// void get_sensor_data (int did, int* result);
sensor_data_t get_sensor_data (int did);

#endif