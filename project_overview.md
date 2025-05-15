# Project Overview

## Project Description
- Project and Scenario Description (e.g. what is the project)

## System Overview
### Hardware Architecture Block Diagram
![alt text](https://github.com/CSSE4011/csse4011-thrasos-pistachio/blob/main/images/block.png)

### Top-level flow chart of software implementation (mote and PC)

## DIKW Pyramid Abstraction
![alt text](https://github.com/CSSE4011/csse4011-thrasos-pistachio/blob/main/images/dikw.png)

## Sensor Integration
* **USB Camera:**
    * **Function:** Capture videos and images of discarded waste items.
    * **Data Required:** Clear image of the object for visual feature extraction and classification.
    * **Interface:** Interfaced with the Jetson Xavier NX.
    * **Protocol:** USB.
    * **Implementation:** Machine learning libraries (e.g. Tensorflow, PyTorch) to classify, convolute and process images.

* **Ultrasonic Distance Sensor:**
    * **Function:** Measure the distance to the top of the waste inside the bin.
    * **Data Required:** Distance reading in centimeters.
    * **Interface:** Digital I/O pins (trigger and echo) with the microcontroller.
    * **Implementation:** Timers will measure the echo pulse duration to determine distance.

* **VOC Sensor (likely integrated into a Temperature, Humidity, and Pressure sensor module):**
    * **Function:** Detect Volatile Organic Compounds, indicative of organic waste.
    * **Data Required:** VOC concentration reading.
    * **Interface:** GPIO pins with the microcontroller.
    * **Implementation:** Specific libraries for the sensor module will be used to read VOC data. <zephyr/drivers/sensor/ccs811.h>

- What sensors are used? What type of data is required? How are the sensors integrated?

## Wireless Network Communication or IoT protocols/Web dashboards 
- What is the network topology or IoT protocols used? What protocols are used and how? What sort of data rate is required? You must also include a message protocol diagram.

## Deliverables and Key Performance Indicators
1.  **Waste Classification Accuracy:**
    * **Target:** Achieve a minimum accuracy of 80% in correctly classifying at least 3 different common waste types (e.g. recycling, general, organic) based on image and potentially VOC data.
    * **Measurement:** \[\frac{\text{Number of correctly classified items}}{\text{Total number of items tested}} \times 100\%\]

2.  **Fill Level Measurement Accuracy:**
    * **Target:** Obtain a fill level measurement accuracy within +/- 10% of the actual fill level of the bin, across at least 80% of the bin's capacity range.
    * **Measurement:** \[\left(1 - \frac{|\text{Sensor Reading} - \text{Actual Level}|}{\text{Total Bin Height}}\right) \times 100\%\]

3.  **Real-time Response Time:**
    * **Target:** The time taken from showing the object to eliciting a response on the servo motor and M5 Core2 display should be less than 1 second for at least 90% of waste disposal. 
    * **Measurement:** Time elapsed from waste detection to feedback display and servo motor movement. 

4.  **Web Dashboard Reliability:**
    * **Target:** Successfully log at least 95% of all waste disposal events (including classification, fill level and timestamp) to the web dashboard.
    * **Measurement:** \[\frac{\text{Number of events logged}}{\text{Total number of disposal events}} \times 100\%\]

5.  **Organic Material Identification Accuracy (VOC Sensor):**
    * **Target:** Achieve a minimum accuracy of 80% in correctly identifying the presence of organic material within the waste deposited, based on the VOC sensor readings.
    * **Measurement:** \[\frac{\text{Number of correctly identified organic/non-organic items (based on VOC)}}{\text{Total number of organic and non-organic items tested}} \times 100\%\]

- at least 5 Deliverables and Key Performance Indicators - how is the ’success’ of the project measured?

## Algorithms Schemes (not assessed in Milestone)
- e.g. Machine learning approaches

## Project Software/Hardware management (not assessed in Milestone)
- Develop a task allocation and timeline and use a novel way of visualising (e.g. Gantt or spiral chart)