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

#include <zephyr/drivers/gpio.h> // For GPIO driver

LOG_MODULE_REGISTER(uart_comm, LOG_LEVEL_DBG);

const struct device *const uart_dev = DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart);

#define RING_BUF_SIZE 1024
uint8_t ring_buffer[RING_BUF_SIZE];
struct ring_buf ringbuf;

//message queue for class id
K_MSGQ_DEFINE(classification_msgq, sizeof(int), 10, 4);

K_SEM_DEFINE(uart_data_sem, 0, 1);

static bool rx_throttled;

#define PROCESS_THREAD_STACK_SIZE 1024
K_THREAD_STACK_DEFINE(process_thread_stack, PROCESS_THREAD_STACK_SIZE);
static struct k_thread process_thread_data;

#define MAX_LINE_LENGTH 128

char line_buffer[MAX_LINE_LENGTH];
size_t line_buffer_pos = 0;

void send_to_jetson(const char* message)
{
    if (!message) return;
    
    int len = strlen(message);
    for (int i = 0; i < len; i++) {
        uart_poll_out(uart_dev, message[i]);
    }
    
    printk("Sent to Jetson: %s", message);
}

void send_classification_result(int class_id)
{
    char msg[32];
    snprintf(msg, sizeof(msg), "CLASS:%d\r\n", class_id);
    send_to_jetson(msg);
}

void send_fill_result(float fill)
{
    char msg[32];
    snprintf(msg, sizeof(msg), "FILL:%.2f\n", (double)fill);
    send_to_jetson(msg);
}

void process_data_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1);
    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    printk("UART Processing Thread started.\n");

    while (1) {
        k_sem_take(&uart_data_sem, K_FOREVER);

        uint8_t byte;
        while (ring_buf_get(&ringbuf, &byte, 1) == 1) {
            // Accumulate bytes, checking for buffer overflow
            if (line_buffer_pos < (MAX_LINE_LENGTH - 1)) {
                line_buffer[line_buffer_pos++] = byte;
            } else {
                printk("Error: Line buffer overflow. Resetting.\n");
                line_buffer_pos = 0; // Discard partial line
                continue;
            }

            if (byte == '\n') {
                line_buffer[line_buffer_pos] = '\0'; 

                if (line_buffer_pos > 0) {
                    line_buffer[line_buffer_pos - 1] = '\0';
                }

                printk("Processing line: \"%s\"\n", line_buffer);

                char* token;

                token = strtok(line_buffer, ",");

                while (token != NULL) {
                    printk("- Class: '%s'\n", token);

                    int class_id = atoi(token);

                    //send_classification_result(class_id);

                    if (k_msgq_put(&classification_msgq, &class_id, K_NO_WAIT) != 0) {
                        printk("Failed to put class_id %d into classification queue\n", class_id);
                    } else {
                        printk("Added class_id %d to classification queue\n", class_id);
                    }

                    token = strtok(NULL, ",");
                }

                line_buffer_pos = 0;
            }
        }

        // Re-enable RX if throttled and space is available
        if (rx_throttled && uart_dev && ring_buf_space_get(&ringbuf) > (MAX_LINE_LENGTH * 2)) {
            uart_irq_rx_enable(uart_dev);
            rx_throttled = false;
        }
    }
}

static void interrupt_handler(const struct device *dev, void *user_data)
{
    ARG_UNUSED(user_data);

    while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {

        if (!rx_throttled && uart_irq_rx_ready(dev)) {
            
            int recv_len;
            uint8_t buffer[64];
            size_t len = MIN(ring_buf_space_get(&ringbuf), sizeof(buffer));

            if (len == 0) {
                /* Throttle because ring buffer is full */
                uart_irq_rx_disable(dev);
                rx_throttled = true;
                printk("Ring buffer full, throttling RX");
                continue;
            }

            recv_len = uart_fifo_read(dev, buffer, len);
            if (recv_len < 0) {
                printk("Failed to read UART FIFO");
                recv_len = 0;
            }

            if (recv_len > 0) {
                int rb_len = ring_buf_put(&ringbuf, buffer, recv_len);
                if (rb_len < recv_len) {
                    printk("Dropped %u bytes in ring buffer", recv_len - rb_len);
                }
                
                printk("UART -> ringbuf: %d bytes", rb_len);

                k_sem_give(&uart_data_sem);
            }
        }
    }
}

int serial_init(void)
{
    //send_to_jetson("NRF:Initializing serial communication\n");
    int ret;

    if (!device_is_ready(uart_dev)) {
        printk("CDC ACM device not ready");
        return -1;
    }

    ret = usb_enable(NULL);
    if (ret != 0) {
        printk("Failed to enable USB: %d", ret);
        return -1;
    }

    ring_buf_init(&ringbuf, sizeof(ring_buffer), ring_buffer);

    printk("Waiting for DTR...");
    
    while (true) {
        uint32_t dtr = 0U;
        uart_line_ctrl_get(uart_dev, UART_LINE_CTRL_DTR, &dtr);
        
        if (dtr) {
            printk("DTR detected - connection established");
            break;
        }
        
        k_sleep(K_MSEC(100));
    }

    /* Set DCD and DSR for proper handshaking */
    ret = uart_line_ctrl_set(uart_dev, UART_LINE_CTRL_DCD, 1);
    if (ret) {
        printk("Failed to set DCD: %d", ret);
    }

    ret = uart_line_ctrl_set(uart_dev, UART_LINE_CTRL_DSR, 1);
    if (ret) {
        printk("Failed to set DSR: %d", ret);
    }

    /* Wait for host to complete setup */
    k_msleep(100);
    
    // // Create data processing thread
    k_thread_create(&process_thread_data, process_thread_stack, PROCESS_THREAD_STACK_SIZE, 
                    process_data_thread, NULL, NULL, NULL, K_PRIO_PREEMPT(7), 0, K_NO_WAIT);

    uart_irq_callback_user_data_set(uart_dev, interrupt_handler, NULL);

    uart_irq_rx_enable(uart_dev);

    printk("Serial communication initialized successfully");
    
    // Send a startup message to let Jetson know we're ready
    k_sleep(K_MSEC(500)); // Give time for connection to stabilize
    send_to_jetson("NRF_READY\r\n");

    return 0;
}

