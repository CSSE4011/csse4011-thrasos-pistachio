# CSSE4011 Prac 2
- Wendi Hao
- 46935016

## Tasks

* **Design Task 1: Real Time Clock (RTC)**
    A library has been created to initialize and interact with the on-chip real time counter of the Thingy:52. See lib/rtc.c

* **Design Task 2: Sensor Interfacing**
    A library utilizes Zephyr threads to interface with the following sensors on the Thingy:52. See lib/thingy.c
    * Temperature
    * Humidity
    * Pressure
    * Gas

* **Design Task 3: Command Line Interface (CLI) Shell**
    A command-line interface (CLI) shell has been implemented using the Zephyr Shell library. 
    * `sensor <DID>`: Reads the value of the sensor corresponding to the DID.
    * `rtc w <time_units>`: Sets the RTC time.
    * `rtc r`: Reads the current RTC time.
    Functionality for these commands relies on the libraries created in Design Tasks 1 and 2.

* **Design Task 4: Continuous Sampling**
    The sample command enables continuous sampling of selected sensors. A button also starts and stops the sampling. See lib/button.c
    * `sample s <DID>`: Starts continuous sampling of the sensor with the given DID at the currently set rate.
    * `sample p <DID>`: Stops continuous sampling of the sensor with the given DID.
    * `sample w <rate>`: Sets the continuous sampling rate to `<rate>` seconds.

* **Design Task 6: RGB LED**
    Updates the LED colour every seond based on gas sensor CO2 values. See lib/led.c
    * blue, cyan, green, yellow, magenta, red

* **Design Task 7: Python GUI**
    Python GUI that displays real time sensor values from json shell outputs. 

## Build/Run instructions
- **Build**: west build -b thingy52/nrf52832 --pristine
- **Flash**: west flash --recover
- **GUI**: python gui.py 
- **rtt**: using RTT viewer