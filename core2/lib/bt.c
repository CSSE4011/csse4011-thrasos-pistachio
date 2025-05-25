#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/sys/printk.h>
#include <string.h>
#include "bt.h"
#include <zephyr/sys/byteorder.h>
#include <math.h>

#define IBEACON_COMPANY_ID      0x004C
#define IBEACON_PROXIMITY_TYPE  0x02
#define IBEACON_DATA_LEN        0x15

K_MSGQ_DEFINE(pos_msgq, sizeof(struct pos_data), 10, 4);

//btcommon.eir_ad.entry.data[2:16] == 16:15:ee:18:6b:01:ec:4b:96:ad:bc:b9:6d:16:6e:66
static const uint8_t TARGET_IBEACON_UUID[16] = {
    0x16, 0x15, 0xee, 0x18, 0x6b, 0x01, 0xec, 0x4b,
    0x96, 0xad, 0xbc, 0xb9, 0x6d, 0x16, 0x6e, 0x66
};

static int parse_ibeacon_data(const uint8_t *ad, size_t ad_len, struct pos_data *ibeacon) {

    if (!ad || ad_len < 25) {
        return -EINVAL;
    }

    for (size_t i = 0; i < ad_len; ) {
        uint8_t len = ad[i++];
        if (len == 0) {
            break;
        }
        if (i + len > ad_len) {
            return -EINVAL;
        }
        uint8_t type = ad[i++];

        if (type == BT_DATA_MANUFACTURER_DATA) {
			//save company id
            uint16_t company_id = (uint16_t)ad[i];

            if (company_id == IBEACON_COMPANY_ID && 
				ad[i + 2] == IBEACON_PROXIMITY_TYPE && 
				ad[i + 3] == IBEACON_DATA_LEN) {

                    const uint8_t *adv_uuid = &ad[i + 4];

                    if (memcmp(adv_uuid, TARGET_IBEACON_UUID, 16) == 0) {

                        float scale_factor = 100.0f;

                        uint16_t major_int = sys_get_be16(&ad[i + 20]);
                        uint16_t minor_int = sys_get_be16(&ad[i + 22]);

                        ibeacon->x = (float)major_int / scale_factor;
                        ibeacon->y = (float)minor_int / scale_factor;

                        return 0;
                    }
            }
        }
        i += len - 1;
    }
    return -ENODATA;
}

static void device_found(const bt_addr_le_t *addr, int8_t rssi, uint8_t type,
	struct net_buf_simple *ad) {

	char le_addr[BT_ADDR_LE_STR_LEN];

    struct pos_data ibeacon;
	int err;

	bt_addr_le_to_str(addr, le_addr, sizeof(le_addr));

    // ibeacon.timestamp = k_uptime_get_32();
	
	err = parse_ibeacon_data(ad->data, ad->len, &ibeacon);

    // printk("err = %d\n", err);

	if (err == 0) {
		//print to check if we received correct data..
        double round_x = round(ibeacon.x);
        double round_y = round(ibeacon.y);

        printk("Discovered: x - %.2f, y - %.2f\n",
            round_x, round_y);
		
		//send over the ibeacon data block
        k_msgq_put(&pos_msgq, &ibeacon, K_NO_WAIT);
	}

}

int observer_start(void) {
	struct bt_le_scan_param scan_param = {
		.type       = BT_LE_SCAN_TYPE_PASSIVE,
        // .type       = BT_LE_SCAN_TYPE_ACTIVE,
		.options    = BT_LE_SCAN_OPT_FILTER_DUPLICATE,
		.interval   = BT_GAP_SCAN_FAST_INTERVAL,
		.window     = BT_GAP_SCAN_FAST_WINDOW,
	};

    int err;

	err = bt_le_scan_start(&scan_param, device_found);

	if (err) {
		printk("Start scanning failed (err %d)\n", err);
		return err;
	}
	printk("Started scanning...\n");

	return 0;
}

void sensor_thread(void *p1, void *p2, void *p3) {
    printk("Starting Sensor Thread (iBeacon Scanner)\n");

    int err;
    err = bt_enable(NULL);
    if (err) {
        printk("Bluetooth init failed (err %d)\n", err);
        return;
    }
    printk("Bluetooth initialized\n");

    observer_start();
    printk("Exiting Sensor Thread.\n");
}

K_THREAD_DEFINE(sensor, 2048, sensor_thread, NULL, NULL, NULL, 7, 0, 0);