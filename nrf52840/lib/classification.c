// Receive VoC bluetooth data from thingy52 
// Receieve image serial USB data from Jetson
// Determine waste classification
// Send to servo/ display

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/bluetooth/hci.h>

K_MSGQ_DEFINE(position_servo_msgq, sizeof(uint8_t), 10, 4);
K_MSGQ_DEFINE(position_disp_msgq, sizeof(uint8_t), 10, 4);

#define IBEACON_COMPANY_ID      0x004C
#define IBEACON_PROXIMITY_TYPE  0x02
#define IBEACON_DATA_LEN        0x15

#define STACKSIZE 1024

volatile float voc_ppb_received = 0.0f;

void receive_voc_thread(void);
K_THREAD_DEFINE(voc_receive_tid, STACKSIZE, receive_voc_thread, NULL, NULL, NULL, 5, 0, 0);

// #define CLASSIFICATION_TEST_PRIORITY 7
// void classification_test_thread(void);

// K_THREAD_DEFINE(classification_test_tid, STACKSIZE,
//     classification_test_thread, NULL, NULL, NULL,
//     CLASSIFICATION_TEST_PRIORITY, 0, 0);

// void classification_test_thread(void) {
//     // Simulated classification results: 0 = non-organic, 1 = organic
//     uint8_t test_positions[] = {0, 1, 1, 0, 1, 1, 0, 0, 1, 1};
//     size_t count = sizeof(test_positions) / sizeof(test_positions[0]);

//     for (size_t i = 0; i < count; i++) {
//         uint8_t pos = test_positions[i];

//         // Send to servo queue
//         k_msgq_put(&position_servo_msgq, &pos, K_NO_WAIT);

//         // Send to display queue
//         k_msgq_put(&position_disp_msgq, &pos, K_NO_WAIT);

//         k_sleep(K_SECONDS(5));  // Simulate time between classifications
//     }


//     // Idle loop
//     while (1) {
//         k_sleep(K_FOREVER);
//     }
// }

static const uint8_t TARGET_UUID[16] = {
    0x16, 0x15, 0xee, 0x18, 0x6b, 0x01, 0xec, 0x4b,
    0x96, 0xad, 0xbc, 0xb9, 0x6d, 0x16, 0x6e, 0x66
};

static int parse_data(const uint8_t *ad, size_t ad_len) {
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

                    if (memcmp(adv_uuid, TARGET_UUID, 16) == 0) {

                        uint16_t adv_major = sys_get_le16(&ad[i + 20]);
                        voc_ppb_received = adv_major / 100.0f;
                    }
            }
        }
        i += len - 1;
    }
    return -ENODATA; // Data not found
}

static void device_found(const bt_addr_le_t *addr, int8_t rssi, uint8_t type,
	struct net_buf_simple *ad) {

	char le_addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(addr, le_addr, sizeof(le_addr));

	err = parse_data(ad->data, ad->len);

	if (err == 0) {
		//print to check if we received correct data..
        printk("Found data, voc = %.2f\n", voc_ppb_received);
	}

}

void receive_voc_thread(void) {
    struct bt_le_scan_param scan_param = {
		// .type       = BT_LE_SCAN_TYPE_PASSIVE,
        .type       = BT_LE_SCAN_TYPE_ACTIVE,
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