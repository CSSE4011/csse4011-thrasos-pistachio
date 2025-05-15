#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/util.h>
#include <zephyr/sys/printk.h>
#include <zephyr/data/json.h>
#include <zephyr/shell/shell.h>
#include "thingy.h"
#include "rtc.h"

#define BUTTON DT_ALIAS(sw0)
static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(BUTTON, gpios);
static struct gpio_callback button_cb_data;

#define STACK_SIZE 1024*2
K_THREAD_STACK_DEFINE(sampling_stack, STACK_SIZE);
static struct k_thread sampling_thread;

#define ALL_STACK_SIZE 1024*4
K_THREAD_STACK_DEFINE(all_sampling_stack, ALL_STACK_SIZE);
static struct k_thread all_sampling_thread;

typedef enum {
    SAMPLING_INACTIVE,
    SAMPLING_RUNNING,
    SAMPLING_SUSPENDED
} sampling_state;

static int sampling_rate = 0;
static int current_did = -1;
static sampling_state state = SAMPLING_INACTIVE;
static sampling_state all_state = SAMPLING_INACTIVE;

//{<DID>,<RTC TIME>,[<DEVICE VALUE>]}

struct sensor_info {
	int did;
	char *time;
	int data[2];
    char *data_str;
	size_t data_len; 
};

//json descriptor for sensor info
static const struct json_obj_descr sensor_descr[] = {
    JSON_OBJ_DESCR_PRIM(struct sensor_info, did, JSON_TOK_NUMBER),
    JSON_OBJ_DESCR_PRIM(struct sensor_info, time, JSON_TOK_STRING),
    // JSON_OBJ_DESCR_ARRAY(struct sensor_info, data, 2, data_len, JSON_TOK_NUMBER)
    JSON_OBJ_DESCR_PRIM(struct sensor_info, data_str, JSON_TOK_STRING)
};

struct all_sensors_info {
    int did;
    struct sensor_info temperature;
    struct sensor_info humidity;
    struct sensor_info pressure;
    struct sensor_info gas;
};

static const struct json_obj_descr all_sensors_descr[] = {
    // JSON_OBJ_DESCR_PRIM(struct all_sensors_info, did, JSON_TOK_NUMBER),
    JSON_OBJ_DESCR_OBJECT(struct all_sensors_info, temperature, sensor_descr),
    JSON_OBJ_DESCR_OBJECT(struct all_sensors_info, humidity, sensor_descr),
    JSON_OBJ_DESCR_OBJECT(struct all_sensors_info, pressure, sensor_descr),
    JSON_OBJ_DESCR_OBJECT(struct all_sensors_info, gas, sensor_descr)
};

void sample_sensor(void *p1, void *p2, void *p3) {
	int rate = (int)p2;
	int did = (int) p1;

    struct sensor_info sensor = {
        .did = did,
        .data_len = 2
    };
    sensor_data_t received_data;
	char output[1024];

	while(1) {
        k_mutex_lock(&sensor_mutex, K_FOREVER);
        
        received_data = get_sensor_data(did);
        k_mutex_unlock(&sensor_mutex);

        sensor.time = k_malloc(strlen(received_data.timestamp) + 1);
        strcpy(sensor.time, received_data.timestamp);
        
        // sensor.data[0] = received_data.sensor.val1;
        // sensor.data[1] = received_data.sensor.val2;
        sensor.data_str = k_malloc(20);
        if (did != 3) {
            sprintf(sensor.data_str, "%d.%2d", received_data.sensor.val1, received_data.sensor.val2);
        } else {
            sprintf(sensor.data_str, "%d, %d", received_data.sensor.val1, received_data.sensor.val2);
        }

		//get info from json struct and put into output string
		json_obj_encode_buf(sensor_descr, ARRAY_SIZE(sensor_descr), &sensor, output, sizeof(output));

		printk("%s\n", output);

        k_free(sensor.time);
        k_free(sensor.data_str);

		k_sleep(K_SECONDS(rate));
	}
}

void sample_all_sensor(void *p1, void *p2, void *p3) {
	int rate = (int)p1;

	struct all_sensors_info all_sensors = {
        .did = ALL_DID,
        .temperature = {.did = TEMP_DID, .data_len = 2},
        .humidity = {.did = HUM_DID, .data_len = 2},
        .pressure = {.did = PRES_DID, .data_len = 2},
        .gas = {.did = GAS_DID, .data_len = 2}
    };

	char output[2048];

    sensor_data_t temp_received_data, hum_received_data, pres_received_data, gas_received_data;

    while(1) {
        // Get data for all sensors
        k_mutex_lock(&sensor_mutex, K_FOREVER);
        
        temp_received_data = get_sensor_data(all_sensors.temperature.did);
        hum_received_data = get_sensor_data(all_sensors.humidity.did);
        pres_received_data = get_sensor_data(all_sensors.pressure.did);
        gas_received_data = get_sensor_data(all_sensors.gas.did);
        k_mutex_unlock(&sensor_mutex);

        all_sensors.temperature.time = k_malloc(strlen(temp_received_data.timestamp) + 1);
        strcpy(all_sensors.temperature.time, temp_received_data.timestamp);
        all_sensors.temperature.data_str = k_malloc(20);
        sprintf(all_sensors.temperature.data_str, "%d.%2d", temp_received_data.sensor.val1, temp_received_data.sensor.val2);
        // all_sensors.temperature.data[0] = temp_received_data.sensor.val1;
        // all_sensors.temperature.data[1] = temp_received_data.sensor.val2;

        all_sensors.humidity.time = k_malloc(strlen(hum_received_data.timestamp) + 1);
        strcpy(all_sensors.humidity.time, hum_received_data.timestamp);
        // all_sensors.humidity.data[0] = hum_received_data.sensor.val1;
        // all_sensors.humidity.data[1] = hum_received_data.sensor.val2;
        all_sensors.humidity.data_str = k_malloc(20);
        sprintf(all_sensors.humidity.data_str, "%d.%2d", hum_received_data.sensor.val1, hum_received_data.sensor.val2);

        all_sensors.pressure.time = k_malloc(strlen(pres_received_data.timestamp) + 1);
        strcpy(all_sensors.pressure.time, pres_received_data.timestamp);
        // all_sensors.pressure.data[0] = pres_received_data.sensor.val1;
        // all_sensors.pressure.data[1] = pres_received_data.sensor.val2;
        all_sensors.pressure.data_str = k_malloc(20);
        sprintf(all_sensors.pressure.data_str, "%d.%2d", pres_received_data.sensor.val1, pres_received_data.sensor.val2);

        all_sensors.gas.time = k_malloc(strlen(gas_received_data.timestamp) + 1);
        strcpy(all_sensors.gas.time, gas_received_data.timestamp);
        // all_sensors.gas.data[0] = gas_received_data.sensor.val1;
        // all_sensors.gas.data[1] = gas_received_data.sensor.val2;
        all_sensors.gas.data_str = k_malloc(20);
        sprintf(all_sensors.gas.data_str, "%d, %d", gas_received_data.sensor.val1, gas_received_data.sensor.val2);
        
        json_obj_encode_buf(all_sensors_descr, ARRAY_SIZE(all_sensors_descr), 
                           &all_sensors, output, sizeof(output));

        printk("%s\n", output);
        k_free(all_sensors.temperature.time);
        k_free(all_sensors.humidity.time);
        k_free(all_sensors.pressure.time);
        k_free(all_sensors.gas.time);
        k_free(all_sensors.temperature.data_str);
        k_free(all_sensors.humidity.data_str);
        k_free(all_sensors.pressure.data_str);
        k_free(all_sensors.gas.data_str);

        k_sleep(K_SECONDS(rate));
    }
}

void thread_kill (int type) {
    if (type == 0) {
        if (state != SAMPLING_INACTIVE) {
            k_thread_abort(&sampling_thread);
            state = SAMPLING_INACTIVE;
        }
    } else {
        if (all_state != SAMPLING_INACTIVE) {
            k_thread_abort(&all_sampling_thread);
            all_state = SAMPLING_INACTIVE;
        }
    }
}

int cmd_sample(const struct shell *shell, size_t argc, char **argv) {
	if (strcmp(argv[1], "w") == 0) {
		//sampling rate in secs
        sampling_rate = atoi(argv[2]);

    } else if (strcmp(argv[1], "s") == 0) {
        if (atoi(argv[2]) == ALL_DID) {
            current_did = ALL_DID;
            if (all_state == SAMPLING_INACTIVE) {
                k_thread_create(&all_sampling_thread, all_sampling_stack, ALL_STACK_SIZE, sample_all_sensor, 
                    (void *)sampling_rate, NULL, NULL, K_PRIO_PREEMPT(6), 0, K_NO_WAIT);
                    all_state = SAMPLING_RUNNING;
                    thread_kill(0);
            } else if (all_state == SAMPLING_SUSPENDED) {
                k_thread_resume(&all_sampling_thread);
                all_state = SAMPLING_RUNNING;
            }
        } else {
            if (current_did != atoi(argv[2])) {
                current_did = atoi(argv[2]);
                if (all_state == SAMPLING_RUNNING || all_state == SAMPLING_SUSPENDED) {
                    thread_kill(0);
                }
                k_thread_create(&sampling_thread, sampling_stack, STACK_SIZE, sample_sensor, 
                    (void *)current_did, (void *)sampling_rate, NULL, K_PRIO_PREEMPT(6), 0, K_NO_WAIT);
                thread_kill(1);
                state = SAMPLING_RUNNING;
            } else {
                if (state == SAMPLING_SUSPENDED) {
                    k_thread_resume(&sampling_thread);
                    state = SAMPLING_RUNNING;
                }
            }
        } 

    } else if (strcmp(argv[1], "p") == 0) {
        if (state == SAMPLING_RUNNING) {
            printk("Sampling paused\n");
            k_thread_suspend(&sampling_thread);
            state = SAMPLING_SUSPENDED;
        }
        if (all_state == SAMPLING_RUNNING) {
            printk("Sampling paused for all did sampling\n");
            k_thread_suspend(&all_sampling_thread);
            all_state = SAMPLING_SUSPENDED;
        }
		
    }
    return 0;
}

void button_pressed(const struct device *dev, struct gpio_callback *cb, uint32_t pins) {
	if (state == SAMPLING_SUSPENDED) {
        printk("Button pressed, resume sampling\n");
        k_thread_resume(&sampling_thread);
        state = SAMPLING_RUNNING;
    } else if (state == SAMPLING_RUNNING) {
        printk("Button pressed, pause sampling\n");
        k_thread_suspend(&sampling_thread);
        state = SAMPLING_SUSPENDED;
    }
    if (all_state == SAMPLING_SUSPENDED) {
        printk("Button pressed, resume sampling\n");
        k_thread_resume(&all_sampling_thread);
        all_state = SAMPLING_RUNNING;
    } else if (all_state == SAMPLING_RUNNING) {
        printk("Button pressed, pause sampling\n");
        k_thread_suspend(&all_sampling_thread);
        all_state = SAMPLING_SUSPENDED;
    }
}

void button_init(void) {
    gpio_pin_configure_dt(&button, GPIO_INPUT);
	//active high, detect rising edge
	gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_RISING);

	gpio_init_callback(&button_cb_data, button_pressed, BIT(button.pin));
	gpio_add_callback_dt(&button, &button_cb_data);
}