#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/display.h>
#include <zephyr/drivers/gpio.h>
#include <lvgl.h>
#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <lvgl_input_device.h>
#include "bt.h"
#include "grid.h"
 
#define LOG_LEVEL CONFIG_LOG_DEFAULT_LEVEL
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(app);
 
 
int main(void) {
    char coord_str[30] = {0};
    const struct device *display_dev;
    lv_obj_t *coord_label;

    display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
    if (!device_is_ready(display_dev)) {
        LOG_ERR("Device not ready, aborting test");
        return 0;
    }
    coord_label = lv_label_create(lv_screen_active());
    lv_obj_align(coord_label, LV_ALIGN_TOP_MID, 0, 0);

    lv_timer_handler();
    display_blanking_off(display_dev);
     struct pos_data received_coords;

    while (1) {
       if (k_msgq_get(&pos_msgq, &received_coords, K_FOREVER) == 0) {
           // printk("X: %.2f, Y: %.2f", received_coords.x, received_coords.y);
           
            sprintf(coord_str, "x: %4.2f, y: %4.2f", received_coords.x, received_coords.y);\
            lv_label_set_text(coord_label, coord_str);
       }
        lv_timer_handler();
       k_sleep(K_MSEC(10));
    }
}