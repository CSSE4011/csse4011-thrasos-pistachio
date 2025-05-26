#ifndef BLUETOOTH_H
#define BLUETOOTH_H

#include <zephyr/kernel.h>
#include <zephyr/types.h> // For uint8_t, uint16_t etc.

struct m5data {
    float fill;
    int class;
};

extern struct k_msgq ibeacon_msgq;

void bt_advertiser_init(void);

#endif // BLUETOOTH_H