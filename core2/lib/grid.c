#include <zephyr/kernel.h>
#include <lvgl.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static lv_obj_t *chart;
static lv_chart_series_t *coordinate_series;

void create_coordinate_chart(void) {
    lv_obj_t *screen = lv_screen_active();
    chart = lv_chart_create(screen);
    lv_obj_set_size(chart, 200, 150); // Adjust size as needed
    lv_obj_align(chart, LV_ALIGN_BOTTOM_LEFT, 20, -20); // Position as needed

    // Set chart type (SCATTER for x, y points)
    lv_chart_set_type(chart, LV_CHART_TYPE_SCATTER);

    // Set axis ranges to sensible values for your coordinate system
     lv_chart_set_range(chart, LV_CHART_AXIS_PRIMARY_X, 0, 200); 
     lv_chart_set_range(chart, LV_CHART_AXIS_PRIMARY_Y, 0, 150);

    // Add a series for the coordinates
    coordinate_series = lv_chart_add_series(chart, lv_palette_main(LV_PALETTE_RED), LV_CHART_AXIS_PRIMARY_Y);
    lv_chart_set_point_count(chart, 2); //keep only 2 points, maybe change to 1 possibly??
}

void update_coordinate_chart(float x, float y) {

    if (chart && coordinate_series) {
        // Directly set the single point in the series
        lv_coord_t chart_y = (lv_coord_t)(x * (150.0f / 3.0f));
        lv_coord_t chart_x = (lv_coord_t)(y * (200.0f / 4.0f));

        // printk("%d, %d\n", chart_x, chart_y);
        lv_chart_set_next_value2(chart, coordinate_series, chart_x, chart_y);
    }
}