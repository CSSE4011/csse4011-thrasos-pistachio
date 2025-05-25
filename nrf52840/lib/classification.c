// Receive VoC bluetooth data from thingy52 
// Receieve image serial USB data from Jetson
// Determine waste classification
// Send to servo/ display

#include <zephyr/kernel.h>
#include <zephyr/device.h>

K_MSGQ_DEFINE(position_servo_msgq, sizeof(uint8_t), 10, 4);
K_MSGQ_DEFINE(position_disp_msgq, sizeof(uint8_t), 10, 4);

#define STACKSIZE 1024
#define CLASSIFICATION_TEST_PRIORITY 5

void classification_test_thread(void);

K_THREAD_DEFINE(classification_test_tid, STACKSIZE,
    classification_test_thread, NULL, NULL, NULL,
    CLASSIFICATION_TEST_PRIORITY, 0, 0);

void classification_test_thread(void) {
    // Simulated classification results: 0 = non-organic, 1 = organic
    uint8_t test_positions[] = {0, 1, 1, 0, 1};
    size_t count = sizeof(test_positions) / sizeof(test_positions[0]);

    for (size_t i = 0; i < count; i++) {
        uint8_t pos = test_positions[i];

        // Send to servo queue
        if (k_msgq_put(&position_servo_msgq, &pos, K_NO_WAIT) == 0) {
            printk("Sent to SERVO queue: %d\n", pos);
        } else {
            printk("Failed to send to SERVO queue\n");
        }

        // Send to display queue
        if (k_msgq_put(&position_disp_msgq, &pos, K_NO_WAIT) == 0) {
            printk("Sent to DISPLAY queue: %d\n", pos);
        } else {
            printk("Failed to send to DISPLAY queue\n");
        }

        k_sleep(K_SECONDS(5));  // Simulate time between classifications
    }

    printk("Classification test complete.\n");

    // Idle loop
    while (1) {
        k_sleep(K_FOREVER);
    }
}