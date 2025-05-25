#include <classification.h>
#include <display_data.h>
#include <servo.h>
#include <serial.h>
#include <ultrasonic.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <bt.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/bluetooth/gap.h>

int main(void) {
    printk("here\n");
    ultrasonic_setup();
    

    // //set up serial code
    serial_init();

     // Initialize Bluetooth
     int err = bt_enable(NULL);
     if (err) {
         printk("Bluetooth init failed (err %d)\n", err);
         return;
     }
 
     printk("Bluetooth initialized\n");

     bt_advertiser_init();
    
    // test();

    return 0;
}