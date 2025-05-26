#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/drivers/gpio.h>
#include <serial.h>
#include <bt.h>
#include <ultrasonic.h>

#define BUTTON_NODE DT_ALIAS(sw0)
static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(BUTTON_NODE, gpios);

K_MSGQ_DEFINE(position_servo_msgq, sizeof(uint8_t), 10, 4);
K_MSGQ_DEFINE(position_disp_msgq, sizeof(uint8_t), 10, 4);

#define IBEACON_COMPANY_ID      0x004C
#define IBEACON_PROXIMITY_TYPE  0x02
#define IBEACON_DATA_LEN        0x15

#define STACKSIZE 1024
K_THREAD_STACK_DEFINE(stack_1, STACKSIZE);
K_THREAD_STACK_DEFINE(stack_2, STACKSIZE);
K_THREAD_STACK_DEFINE(stack_3, STACKSIZE);

#define VOC_THRESHOLD 50.0f

volatile float voc_ppb_received = 0.0f;
volatile uint8_t last_processed_class = 0;

void receive_voc_thread(void *p1, void *p2, void *p3);
void button_thread(void *p1, void *p2, void *p3);
void receive_classification_thread(void *p1, void *p2, void *p3);

static struct k_thread receive_voc_thread_data;
static struct k_thread button_thread_data;
static struct k_thread receive_classification_thread_data;


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

void receive_voc_thread(void *p1, void *p2, void *p3) {
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

uint8_t process_class(uint8_t item_number) {

    if (item_number >= 46 && item_number <= 55) {
        return 0; // Organic
    }

    // Non-organic: bottles, utensils, bowls, appliances
    if ((item_number >= 39 && item_number <= 45) || 
        (item_number >= 70 && item_number <= 72)) {
        return 1; // Non-organic
    }

    if (voc_ppb_received > VOC_THRESHOLD) {
        send_to_jetson("voc reached\n");
        return 0;
    } else {
        return 1;
    }

    // return 1;
}

void receive_classification_thread(void *p1, void *p2, void *p3) {
    uint8_t item_number;
    struct m5data sending_data;
    float fill_level;

    while(1) {
        if (k_msgq_get(&classification_msgq, &item_number, K_FOREVER) == 0) { // Receive position message
            last_processed_class = process_class(item_number);
            printk("processed class = %d\n", last_processed_class);

            send_classification_result(last_processed_class);
            
            // Send to display queue
            k_msgq_put(&position_disp_msgq, &last_processed_class, K_NO_WAIT);

            if (k_msgq_get(&fill_level_msgq, &fill_level, K_FOREVER) == 0) { // Receive position message
                send_fill_result(fill_level);
                sending_data.fill = fill_level;
            }

            sending_data.class = last_processed_class;
            //send to bt queue
            if(k_msgq_num_free_get(&ibeacon_msgq) == 0) {
                struct m5data dummy;
                k_msgq_get(&ibeacon_msgq, &dummy, K_NO_WAIT);
            }
            k_msgq_put(&ibeacon_msgq, &sending_data, K_NO_WAIT);
        }
    }
}

void button_thread(void *p1, void *p2, void *p3) {
    bool button_pressed;
    bool last_state = false;

    if (!device_is_ready(button.port)) {
        printk("Button device not ready!\n");
        return;
    }

    gpio_pin_configure_dt(&button, GPIO_INPUT);

    while (1) {
        button_pressed = gpio_pin_get_dt(&button);

        if (button_pressed && !last_state) {
            // Rising edge: button just pressed
            k_msgq_put(&position_servo_msgq, &last_processed_class, K_NO_WAIT);
        }

        last_state = button_pressed;
        k_msleep(1000); // debounce delay
    }
}

void threads_init(void) {
    k_thread_create(&receive_voc_thread_data, stack_1, STACKSIZE, receive_voc_thread, NULL, NULL, NULL, K_PRIO_PREEMPT(7), 0, K_NO_WAIT);
    k_thread_create(&button_thread_data, stack_2, STACKSIZE, button_thread, NULL, NULL, NULL, K_PRIO_PREEMPT(6), 0, K_NO_WAIT);
    k_thread_create(&receive_classification_thread_data, stack_3, STACKSIZE, receive_classification_thread, NULL, NULL, NULL, K_PRIO_PREEMPT(7), 0, K_NO_WAIT);
}