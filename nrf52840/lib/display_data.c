// Receive ultrasonic data
// Receive classification data
// Send data to M5 Stack Core (Hayden)

#include "classification.h"
#include "ultrasonic.h"
#include "stdint.h"
#include "zephyr/kernel.h"

#define STACKSIZE 1024
#define RECEIVE_CLASSIFICATION_PRIORITY 7
#define RECEIVE_FILL_LEVEL_PRIORITY 7

void receive_classification_thread(void);
void receive_fill_level_thread(void);

K_THREAD_DEFINE(receive_classification_tid, STACKSIZE, receive_classification_thread, NULL, NULL, NULL, RECEIVE_CLASSIFICATION_PRIORITY, 0, 0);
K_THREAD_DEFINE(receive_fill_level_tid, STACKSIZE, receive_fill_level_thread, NULL, NULL, NULL, RECEIVE_FILL_LEVEL_PRIORITY, 0, 0);

void receive_classification_thread(void) {
    uint8_t pos;

    while(1) {
        if (k_msgq_get(&position_disp_msgq, &pos, K_FOREVER) == 0) { // Receive position message
            printk("position = %d\n", pos);
        }
    }
}

void receive_fill_level_thread(void) {
    float fill_level;

    while(1) {
        if (k_msgq_get(&fill_level_msgq, &fill_level, K_FOREVER) == 0) { // Receive position message
            printk("fill level = %.2f\n", (double)fill_level);
        }
    }
}