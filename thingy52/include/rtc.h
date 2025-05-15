#ifndef RTC_LIB
#define RTC_LIB 

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/rtc.h>
#include <zephyr/sys/util.h>
#include <zephyr/shell/shell.h>
#include <zephyr/drivers/counter.h>
#include <SEGGER_RTT.h>

typedef struct {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
} date_time_t;

void rtc_init(void);

void set_date_time(const date_time_t *new_time);
// char* get_date_time(date_time_t *current_time);
void get_date_time(char *buffer, size_t size);
int cmd_rtc(const struct shell *shell, size_t argc, char **argv);
// void date_to_str (char *buffer, size_t size, date_time_t time);

#endif