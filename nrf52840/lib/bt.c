#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/sys/printk.h>
#include <string.h>

// Define the iBeacon data with hardcoded values
static uint8_t data_ibeacon[] = {
    0x4C, 0x00,             // Apple Company ID (Little Endian)
    0x02, 0x15,             // iBeacon Proximity Type and Data Length
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, // Hardcoded UUID
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x00, 0x01,             // Hardcoded Major Value (0x0001)
    0x00, 0x02,             // Hardcoded Minor Value (0x0002)
    0xC8                    // Hardcoded Calibrated TX Power (-56 dBm)
};

static struct bt_data ad[] = {
    BT_DATA(BT_DATA_FLAGS, (uint8_t *)BT_LE_AD_NO_BREDR, 1),
    BT_DATA(BT_DATA_MANUFACTURER_DATA, data_ibeacon, sizeof(data_ibeacon)),
};


void bt_send(void)
{

        err = bt_le_adv_start(BT_LE_ADV_NCONN, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
        if (err) {
            printk("Advertising failed to start (err %d)\n", err);
        } 
		k_sleep(K_SECONDS(1));
}