#ifndef BLUETOOTH_H
#define BLUETOOTH_H

#include <zephyr/kernel.h>
#include <zephyr/types.h> // For uint8_t, uint16_t etc.

typedef struct {
    uint16_t major; // Will store the class ID (0-79)
    uint16_t minor; // Can be used for instance ID or fixed value like 1
} ibeacon_data_t;


extern struct k_msgq ibeacon_msgq;

void bt_advertiser_init(void);

#endif // BLUETOOTH_H