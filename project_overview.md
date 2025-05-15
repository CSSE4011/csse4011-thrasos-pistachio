# Project Overview

## Project Description
## Smart Waste Management System

A smart waste bin system designed to encourage correct waste sorting and track waste generation using embedded sensors and real-time feedback.

**Smart Bin Components:**

* **Camera:** Captures images of discarded items for waste type classification (e.g., general, recycling, organic).
* **Ultrasonic Sensor:** Measures the fill level of the bin.
* **VoC Sensor:** Detects volatile organic compounds to aid in identifying organic material.
* **M5 Core:** Provides a real-time visual display for user feedback.
* **Servo Motor:** Guides waste into the correct compartment.

**Waste Detection and Processing:**

1.  **Waste Detected:** When an item is presented to the bin, the sensors are triggered.
2.  **Image Analysis:** The camera captures an image of the waste, which is then analyzed to classify the waste type (e.g., paper, plastic, organic).
3.  **Fill Level Recording:** The ultrasonic sensor measures and records the current fill level of the bin.
4.  **Real-time Visual Feedback:** The M5 Core displays visual information to the user, indicating the correct bin compartment for the identified waste type.
5.  **Servo Motor Guidance:** Based on the waste classification, the servo motor moves to guide the waste into the appropriate bin compartment.
6.  **Data Transmission:** The identified waste type and the bin's fill level are transmitted wirelessly to a web dashboard.

**Web Dashboard Features:**

* **Waste History:** Displays a chronological record of discarded waste, including the identified waste type and the timestamp of disposal.
* **Current Fill Level:** Shows the real-time fill level of the smart bin.
* **Sorting Accuracy Statistics:** Provides data and visualizations on the user's waste sorting over time. 

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
- MQTT protocol via Wifi, local access point setup, through TBD broker to TAGIO unless decision made to include camera feed. Uses proposed message protocol in any communication instance

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