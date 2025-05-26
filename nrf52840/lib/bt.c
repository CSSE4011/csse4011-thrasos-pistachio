#include <bt.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/bluetooth/gap.h>
#include <string.h> // For memcpy
#include <zephyr/sys/printk.h>

#define BT_STACK_SIZE 1024
K_THREAD_STACK_DEFINE(bt_advertiser_thread_stack, BT_STACK_SIZE);
static struct k_thread bt_advertiser_thread_data;

K_MSGQ_DEFINE(ibeacon_msgq, sizeof(struct m5data), 5, 4);

static const uint8_t BASE_IBEACON_UUID[16] = {
    0x16, 0x15, 0xee, 0x18, 0x6b, 0x01, 0xec, 0x4b,
    0x96, 0xad, 0xbc, 0xb9, 0x6d, 0x16, 0x6e, 0x69
};

static uint8_t current_ibeacon_adv_data[25];

static struct bt_data ad[2];

void bt_advertiser_thread(void *p1, void *p2, void *p3) {
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    printk("Bluetooth Advertiser Thread started.\n");

    struct m5data received;
    int err;

    ad[0].type = BT_DATA_FLAGS;
	ad[0].data = (uint8_t *)BT_LE_AD_NO_BREDR;
	ad[0].data_len = 1;
	
	ad[1].type = BT_DATA_MANUFACTURER_DATA;
	ad[1].data = current_ibeacon_adv_data;
	ad[1].data_len = sizeof(current_ibeacon_adv_data);

    current_ibeacon_adv_data[0] = 0x4C; // Apple Company ID (LSB)
    current_ibeacon_adv_data[1] = 0x00; // Apple Company ID (MSB)
    current_ibeacon_adv_data[2] = 0x02; // iBeacon Proximity Type
    current_ibeacon_adv_data[3] = 0x15; // iBeacon Data Length (21 bytes following)

    while (1) {
        // k_msleep(500);
        // Wait indefinitely for new iBeacon data from the message queue
        if (k_msgq_get(&ibeacon_msgq, &received, K_FOREVER) == 0) {
            printk("Advertiser received new iBeacon data from queue.\n");

            // 1. Stop current advertising (if any)
            err = bt_le_adv_stop();
            if (err && err != -EALREADY) {
                printk("Advertising failed to stop (err %d)\n", err);
            } else if (err == 0) {
                printk("Previous advertising stopped.\n");
            }

            memcpy(&current_ibeacon_adv_data[4], BASE_IBEACON_UUID, 16);

            // For the float 'fill_value', scale and cast to uint16_t
            // max fill_value is 655.35
            uint16_t major_value_encoded = (uint16_t)(received.fill * 100.0f);
            
            // For the int 'class' (0 or 1), cast to uint16_t
            uint16_t minor_value_encoded = (uint16_t)received.class;

            sys_put_be16(major_value_encoded, &current_ibeacon_adv_data[20]); // Put encoded Major
            sys_put_be16(minor_value_encoded, &current_ibeacon_adv_data[22]);

            current_ibeacon_adv_data[24] = 0xC8;

            err = bt_le_adv_start(BT_LE_ADV_NCONN, ad, ARRAY_SIZE(ad), NULL, 0);
            if (err) {
                printk("Advertising failed to start (err %d)\n", err);
            } else {
                printk("Advertising updated iBeacon (Fill(Major): %u, Class(Minor): %u).\n", received);

                k_sleep(K_MSEC(500)); // Advertise this beacon for 500ms
            }
        } else {
            printk("Failed to get data from iBeacon message queue.\n");
        }
    }
}

void bt_advertiser_init(void) {
    k_thread_create(&bt_advertiser_thread_data, bt_advertiser_thread_stack, BT_STACK_SIZE,
                    bt_advertiser_thread, NULL, NULL, NULL,
                    K_PRIO_PREEMPT(6), 0, K_NO_WAIT);
}