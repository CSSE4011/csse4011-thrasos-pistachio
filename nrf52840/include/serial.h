#ifndef SERIAL_RECEIVER_H
#define SERIAL_RECEIVER_H

extern struct k_msgq uart_rx_msgq;

void process_data_thread(void *p1, void *p2, void *p3);

int serial_init (void);

#endif