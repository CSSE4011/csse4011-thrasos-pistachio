#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/rtc.h>
#include <zephyr/sys/util.h>
#include <zephyr/shell/shell.h>
#include <zephyr/drivers/counter.h>
#include <SEGGER_RTT.h>

// const struct device *const rtc = DEVICE_DT_GET(DT_NODELABEL(rtc2));
// const struct device *const rtc = DEVICE_DT_GET(DT_ALIAS(rtc));
const struct device *const dev_rtc = DEVICE_DT_GET(DT_NODELABEL(rtc2));

typedef struct {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
} date_time_t;

date_time_t base_time;
int base; 

const int days_in_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

// Leap year check
int is_leap_year(int year) {
    return (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}

void add_s_date_time(date_time_t *dt, uint64_t s_to_add) {
    dt->second += s_to_add % 60;
    s_to_add /= 60;
    
    //sec overflow
    if (dt->second >= 60) {
        dt->second -= 60;
        s_to_add++;
    }

    dt->minute += s_to_add % 60;
    s_to_add /= 60;
    
    //min overflow
    if (dt->minute >= 60) {
        dt->minute -= 60;
        s_to_add++;
    }

    dt->hour += s_to_add % 24;
    s_to_add /= 24;
    
    // hour overflow
    if (dt->hour >= 24) {
        dt->hour -= 24;
        s_to_add++;
    }

    while (s_to_add > 0) {
        int month_days = days_in_month[dt->month - 1];

        // Feb
        if (dt->month == 2 && is_leap_year(dt->year)) {
            month_days = 29;
        }

        int days_left = month_days - dt->day;
        
        if (s_to_add <= days_left) {
            //add days to month
            dt->day += s_to_add;
            s_to_add = 0;
        } else {
            //need to go to next month
            s_to_add -= (days_left + 1);
            dt->day = 1;
            dt->month++;
            
            if (dt->month > 12) {
                dt->month = 1;
                dt->year++;
            }
        }
    }
}

void rtc_init(void) {
    if (!device_is_ready(dev_rtc)) {
        printk("RTC device not ready\n");
        return;
    }

    counter_start(dev_rtc);
    int ticks;
    counter_get_value(dev_rtc, &ticks);
    base = ticks / counter_get_frequency(dev_rtc);

    base_time = (date_time_t){
        .year = 0,
        .month = 0,
        .day = 0,
        .hour = 0,
        .minute = 0,
        .second = 0
    };
}

void set_date_time(const date_time_t *new_time)
{
    //save new_time to base_time 
    base_time = *new_time;
    
    int ticks;
    counter_get_value(dev_rtc, &ticks);

    //set base to the current device counter time
    base = ticks / counter_get_frequency(dev_rtc);
}

void get_date_time(char *buffer, size_t size) {

    int current_ticks;
    //get current device counter ticks
    counter_get_value(dev_rtc, &current_ticks);
    
    //calculated # of seconds past time at set 
    int elapsed_seconds = current_ticks / counter_get_frequency(dev_rtc) - base;

    // printk("seconds: %d\n", elapsed_seconds);

    date_time_t current = base_time;
    add_s_date_time(&current, elapsed_seconds);

    snprintf(buffer, size, "%04d-%02d-%02d %02d:%02d:%02d", 
        (int)current.year, (int)current.month, (int)current.day, 
        (int)current.hour, (int)current.minute, (int)current.second);
}

int cmd_rtc(const struct shell *shell, size_t argc, char **argv) {
	int year, month, day, hour, minute, second;

	if (argc > 2 && strcmp(argv[1], "w") == 0) {
		sscanf(argv[2], "%d-%d-%d", &year, &month, &day);
		sscanf(argv[3], "%d:%d:%d", &hour, &minute, &second);

        date_time_t time = {
            .year = year,
            .month = month,
            .day = day,
            .hour = hour,
            .minute = minute,
            .second = second
        };

		set_date_time(&time);
		
	} else if (strcmp(argv[1], "r") == 0) {
        // date_time_t current;
		// get_date_time(&current);

        char buf[32];
        get_date_time(buf, sizeof(buf));
        // date_to_str(buf, sizeof(buf), current);

        printk("Current RTC time: %s\n", buf);
	}
	return 0;
}