// #include <sample_usbd.h>

#include <stdio.h>
#include <string.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/ring_buffer.h>

#include <zephyr/usb/usb_device.h>
#include <zephyr/usb/usbd.h>
#include <zephyr/logging/log.h>
#include <stdlib.h>
#include "bt.h"

const struct device *const uart_dev = DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart);

#define RING_BUF_SIZE 1024
uint8_t ring_buffer[RING_BUF_SIZE];
struct ring_buf ringbuf;

#define MSG_Q_SIZE 10 //10 items
#define MSG_SIZE 256  //256 bytes
//create received serial uart message queue
K_MSGQ_DEFINE(uart_rx_msgq, MSG_SIZE, MSG_Q_SIZE, 4);

//message queue for class id
K_MSGQ_DEFINE(classification_msgq, sizeof(int), 10, 4);

static bool rx_throttled;

#define PROCESS_THREAD_STACK_SIZE 1024
K_THREAD_STACK_DEFINE(process_thread_stack, PROCESS_THREAD_STACK_SIZE);
static struct k_thread process_thread_data;

void process_data_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1);
    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    uint8_t received_message_buffer[MSG_SIZE];

    printk("Data processing thread started.");

    while (true) {
        // Get data from the message queue (blocking call)
        if (k_msgq_get(&uart_rx_msgq, received_message_buffer, K_FOREVER) == 0) {

            if (strcmp(received_message_buffer, "NONE") == 0) {
                printk("No objects detected (received: NONE).");
                continue;
            }

            char* token;
            char* parsable_message = received_message_buffer;

            printk("Detected classes:");

            // Use strtok to split the string
            token = strtok(parsable_message, ",");

            while (token != NULL) {

                printk("- Class: '%s'", token);

                int class_id = atoi(token);

                //need to send classId
                if (class_id) {
                    ibeacon_data_t beacon_to_send;

                    beacon_to_send.major = (uint16_t)class_id;
                    beacon_to_send.minor = 1;
                    
                    //drop old ibeacon
                    if (k_msgq_num_free_get(&ibeacon_msgq) == 0) {
                        ibeacon_data_t dummy_ibeacon;
                        k_msgq_get(&ibeacon_msgq, &dummy_ibeacon, K_NO_WAIT);
                    }

                    k_msgq_put(&ibeacon_msgq, &beacon_to_send, K_NO_WAIT);

                    // Send the class id to classifier
                    k_msgq_put(&classification_msgq, &class_id, K_NO_WAIT);
                }

                // Get the next token
                token = strtok(NULL, ",");
            }
        } else {
            printk("Failed to get data from message queue.");
        }
    }
}

static void interrupt_handler(const struct device *dev, void *user_data)
{
    ARG_UNUSED(user_data);

    while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
        if (!rx_throttled && uart_irq_rx_ready(dev)) {
            int recv_len, rb_len;
            uint8_t buffer[64];
            size_t len = MIN(ring_buf_space_get(&ringbuf),
                    sizeof(buffer));

            if (len == 0) {
                /* Throttle because ring buffer is full */
                uart_irq_rx_disable(dev);
                rx_throttled = true;
                continue;
            }

            recv_len = uart_fifo_read(dev, buffer, len);
            if (recv_len < 0) {
                printk("Failed to read UART FIFO");
                recv_len = 0;
            };

            rb_len = ring_buf_put(&ringbuf, buffer, recv_len);
            if (rb_len < recv_len) {
                printk("Drop %u bytes", recv_len - rb_len);
            }

            printk("tty fifo -> ringbuf %d bytes", rb_len);

            if (rb_len) {
                buffer[rb_len] = '\0';

                if (k_msgq_num_free_get(&uart_rx_msgq) == 0) {
                    char dummy_buf[MSG_SIZE]; 
                    //drop oldest message
                    k_msgq_get(&uart_rx_msgq, dummy_buf, K_NO_WAIT);
                }

                if (k_msgq_put(&uart_rx_msgq, buffer, K_NO_WAIT) == 0) {
                    printk("Sent %d bytes to message queue", rb_len);
                }
            }

        }
    }
}

int serial_init(void)
{
    printk("serial made it");
    int ret;

    if (!device_is_ready(uart_dev)) {
        printk("CDC ACM device not ready");
        return 0;
    }

    ret = usb_enable(NULL);

    if (ret != 0) {
        printk("Failed to enable USB");
        return 0;
    }

    ring_buf_init(&ringbuf, sizeof(ring_buffer), ring_buffer);

    printk("Wait for DTR");

    while (true) {
        uint32_t dtr = 0U;

        uart_line_ctrl_get(uart_dev, UART_LINE_CTRL_DTR, &dtr);
        if (dtr) {
            break;
        } else {
            /* Give CPU resources to low priority threads. */
            k_sleep(K_MSEC(100));
        }
    }

    printk("DTR set");

    /* They are optional, we use them to test the interrupt endpoint */
    ret = uart_line_ctrl_set(uart_dev, UART_LINE_CTRL_DCD, 1);
    if (ret) {
        printk("Failed to set DCD, ret code %d", ret);
    }

    ret = uart_line_ctrl_set(uart_dev, UART_LINE_CTRL_DSR, 1);
    if (ret) {
        printk("Failed to set DSR, ret code %d", ret);
    }

    /* Wait 100ms for the host to do all settings */
    k_msleep(100);

    //create data processing thread
    k_thread_create(&process_thread_data, process_thread_stack, PROCESS_THREAD_STACK_SIZE, process_data_thread, 
        NULL, NULL, NULL, K_PRIO_PREEMPT(7), 0, K_NO_WAIT);

    uart_irq_callback_set(uart_dev, interrupt_handler);

    /* Enable rx interrupts */
    uart_irq_rx_enable(uart_dev);

    return 0;
}