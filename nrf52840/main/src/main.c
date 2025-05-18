#include <classification.h>
#include <display_data.h>
#include <servo.h>
#include <ultrasonic.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>

int main(void) {
    initialise_pins();
    test();
}