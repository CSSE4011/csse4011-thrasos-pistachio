#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/sensor/ccs811.h>
#include <stdio.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/sys/util.h>
#include "thingy.h"
#include "rtc.h"

//set up ring buffers for all data
struct ring_buf gas_ring_1;
struct ring_buf gas_ring_2;
struct ring_buf temp_ring;
struct ring_buf hum_ring;
struct ring_buf pres_ring;

const struct device *const dev_pres = DEVICE_DT_GET_ONE(st_lps22hb_press);
const struct device *const dev_temp = DEVICE_DT_GET_ONE(st_hts221);
const struct device *const dev_gas = DEVICE_DT_GET_ONE(ams_ccs811);


#define DATA_ITEM_SIZE sizeof(sensor_data_t)
#define TIME_STR_SIZE 64

struct k_mutex sensor_mutex;


void gas_read_thread(void *p1, void *p2, void *p3) {
    uint8_t ring_buffer_1[BUFFER_SIZE * DATA_ITEM_SIZE];
    uint8_t ring_buffer_2[BUFFER_SIZE * DATA_ITEM_SIZE];

    ring_buf_init(&gas_ring_1, sizeof(ring_buffer_1), ring_buffer_1);
    ring_buf_init(&gas_ring_2, sizeof(ring_buffer_2), ring_buffer_2);

    while (1) {

        sensor_data_t co2, tvoc;
        
        // Get data
        sensor_sample_fetch(dev_gas);
        sensor_channel_get(dev_gas, SENSOR_CHAN_CO2, &co2.sensor);
        sensor_channel_get(dev_gas, SENSOR_CHAN_VOC, &tvoc.sensor);

        // Get timestamp at point of data reading
        get_date_time(co2.timestamp, sizeof(co2.timestamp));
        get_date_time(tvoc.timestamp, sizeof(tvoc.timestamp));

        ring_buf_put(&gas_ring_1, (uint8_t *)&co2, DATA_ITEM_SIZE);

        ring_buf_put(&gas_ring_2, (uint8_t *)&tvoc, DATA_ITEM_SIZE);

        k_msleep(300); //sleep for 2 second
    }
}

void temp_hum_read_thread(void *p1, void *p2, void *p3) {
    //setting up ring buffer for gas sensor data
    uint8_t temp_ring_data[BUFFER_SIZE * DATA_ITEM_SIZE];
    uint8_t hum_ring_data[BUFFER_SIZE * DATA_ITEM_SIZE];

    ring_buf_init(&temp_ring, sizeof(temp_ring_data), temp_ring_data);
    ring_buf_init(&hum_ring, sizeof(hum_ring_data), hum_ring_data);

    while (1) {

        sensor_data_t temp, hum;
        
        // Get data
        sensor_sample_fetch(dev_temp);
        sensor_channel_get(dev_temp, SENSOR_CHAN_AMBIENT_TEMP, &temp.sensor);
        sensor_channel_get(dev_temp, SENSOR_CHAN_HUMIDITY, &hum.sensor);

        get_date_time(temp.timestamp, sizeof(temp.timestamp));
        get_date_time(hum.timestamp, sizeof(hum.timestamp));

        ring_buf_put(&temp_ring, (uint8_t *)&temp, DATA_ITEM_SIZE);

        ring_buf_put(&hum_ring, (uint8_t *)&hum, DATA_ITEM_SIZE);

        k_msleep(1000); //sleep for 2 second
    }
}

void pressure_read_thread(void *p1, void *p2, void *p3) {
    uint8_t ring_data[BUFFER_SIZE * DATA_ITEM_SIZE];

    //setting up ring buffer for gas sensor data
    ring_buf_init(&pres_ring, sizeof(ring_data), ring_data);

    while (1) {

        sensor_data_t pres;
        
        // Get data
        sensor_sample_fetch(dev_pres);
        sensor_channel_get(dev_pres, SENSOR_CHAN_PRESS, &pres.sensor);

        // Get timestamp at point of data reading
        get_date_time(pres.timestamp, sizeof(pres.timestamp));

        ring_buf_put(&pres_ring, (uint8_t *)&pres, DATA_ITEM_SIZE);

        k_msleep(1000); //sleep for 1 second
    }
}

sensor_data_t get_latest(struct ring_buf *rb) {
    sensor_data_t latest_data;
    memset(&latest_data, 0, sizeof(latest_data));

    while (!ring_buf_is_empty(rb)) {
        ring_buf_get(rb, (uint8_t *)&latest_data, DATA_ITEM_SIZE);
        // The last successful get will overwrite latest_data
    }
    return latest_data;
}

sensor_data_t get_sensor_data (int did){

	// struct sensor_value received_data;
    sensor_data_t received_data;
	// struct ccs811_result_type received_gas;

	switch (did) {
		case TEMP_DID:
            return get_latest(&temp_ring);

		case HUM_DID:
            return get_latest(&hum_ring);
			// ring_buf_get(&hum_ring, (uint8_t *)&received_data, DATA_ITEM_SIZE);

		case PRES_DID:
            return get_latest(&pres_ring);

		case GAS_DID:
            //set up gas data as [co2, tvoc]
            int co2_data = 0;
            received_data = get_latest(&gas_ring_1);
            co2_data = received_data.sensor.val1;

            received_data = get_latest(&gas_ring_2);
            received_data.sensor.val2 = received_data.sensor.val1;
            received_data.sensor.val1 = co2_data;

            return received_data;
			
        default:
            printk("Invalid DID: %d\n", did);
            sensor_data_t empty_data = {0}; 
            return empty_data;
	}
}

K_THREAD_DEFINE(gas_sensor, 2048, gas_read_thread, NULL, NULL, NULL, 7, 0, 0);
K_THREAD_DEFINE(temp_sensor, 2048, temp_hum_read_thread, NULL, NULL, NULL, 7, 0, 0);
K_THREAD_DEFINE(pressure_sensor, 2048, pressure_read_thread, NULL, NULL, NULL, 7, 0, 0);
