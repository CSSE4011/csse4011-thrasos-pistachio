#include <classification.h>
#include <display_data.h>
#include <servo.h>
//#include <serial.h>
#include <ultrasonic.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>

int main(void) {
    printk("here\n");
    ultrasonic_setup();
    

    // //set up serial code
    // serial_init();
    
    // test();
}