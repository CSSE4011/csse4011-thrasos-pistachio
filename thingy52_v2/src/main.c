#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/sensor/ccs811.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <math.h>
#include <string.h>

#define VOC_READ_PRIORITY 6
#define NRF_SEND_PRIORITY 7
#define STACKSIZE 1024

const struct device *const dev_gas = DEVICE_DT_GET_ONE(ams_ccs811);

void voc_read_thread(void);
void send_to_nrf_thread(void);

K_THREAD_DEFINE(voc_read_tid, STACKSIZE, voc_read_thread, NULL, NULL, NULL, VOC_READ_PRIORITY, 0, 0);
K_THREAD_DEFINE(nrf_send_tid, STACKSIZE, send_to_nrf_thread, NULL, NULL, NULL, NRF_SEND_PRIORITY, 0, 0);

K_MSGQ_DEFINE(voc_msgq, sizeof(double), 10, 4);

static struct bt_data ad[2];
static uint8_t data_ibeacon[25];
static const uint8_t custom_uuid[16] = {
    0x16, 0x15, 0xee, 0x18, 0x6b, 0x01, 0xec, 0x4b,
    0x96, 0xad, 0xbc, 0xb9, 0x6d, 0x16, 0x6e, 0x66
};

void voc_read_thread(void) {

    while (1) {
        struct sensor_value tvoc;
        
        // Get data
        sensor_sample_fetch(dev_gas);
        sensor_channel_get(dev_gas, SENSOR_CHAN_VOC, &tvoc);

        double voc_ppb = sensor_value_to_double(&tvoc);
        k_msgq_put(&voc_msgq, &voc_ppb, K_NO_WAIT);

        k_msleep(100);
    }
}

void send_to_nrf_thread(void) {
    double voc_ppb;

    while (1) {
        // Wait for new VOC data
        if (k_msgq_get(&voc_msgq, &voc_ppb, K_FOREVER) == 0) {

            int errn = bt_le_adv_stop();
            if (errn && errn != -EALREADY) {
                printk("Advertising stop failed (err %d)\n", errn);
            }

            data_ibeacon[0] = 0x4C;
            data_ibeacon[1] = 0x00;
            data_ibeacon[2] = 0x02;
            data_ibeacon[3] = 0x15;

            memcpy(&data_ibeacon[4], custom_uuid, 16);

            float scale_factor = 100.0f;
            uint16_t voc_int = (uint16_t)roundf(voc_ppb * scale_factor);

            sys_put_be16(voc_int, &data_ibeacon[20]); // VOC in major
            sys_put_be16(0x0000, &data_ibeacon[22]);

            data_ibeacon[24] = 0xC8;

            ad[0].type = BT_DATA_FLAGS;
            ad[0].data = (uint8_t *)BT_LE_AD_NO_BREDR;
            ad[0].data_len = 1;

            ad[1].type = BT_DATA_MANUFACTURER_DATA;
            ad[1].data = data_ibeacon;
            ad[1].data_len = sizeof(data_ibeacon);

            errn = bt_le_adv_start(BT_LE_ADV_NCONN, ad, ARRAY_SIZE(ad), NULL, 0);
            if (errn) {
                printk("Advertising start failed (err %d)\n", errn);
            } else {
                printk("Advertising VOC: %.2f ppb\n", voc_ppb);
            }

            k_msleep(300);
        }
    }
}

int main(void) {
    int err = bt_enable(NULL);
    if (err) {
        printk("Bluetooth init failed (err %d)\n", err);
        return;
    }

    printk("Bluetooth enabled. Starting threads...\n");
}