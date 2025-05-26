#include <mosquitto.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <ctime>

// MQTT connection parameters
const std::string MQTT_BROKER = "ve179623.ala.asia-southeast1.emqxsl.com";
const int MQTT_PORT = 8883;
const std::string USERNAME = "jetson";
const std::string PASSWORD = "jetson";
const std::string CLIENT_ID = "test_client_" + std::to_string(time(nullptr));
const std::string TEST_TOPIC = "/device/jetson/123123123";

struct mosquitto *mosq = nullptr;
bool connected = false;

// Callback functions
void on_connect(struct mosquitto *mosq, void *userdata, int result) {
    std::cout << "\n=== CONNECTION CALLBACK ===" << std::endl;
    if (result == 0) {
        std::cout << "✓ Successfully connected to MQTT broker!" << std::endl;
        connected = true;
    } else {
        std::cout << "✗ Connection failed with code: " << result << std::endl;
        switch(result) {
            case 1: std::cout << "  -> Connection refused: unacceptable protocol version" << std::endl; break;
            case 2: std::cout << "  -> Connection refused: identifier rejected" << std::endl; break;
            case 3: std::cout << "  -> Connection refused: server unavailable" << std::endl; break;
            case 4: std::cout << "  -> Connection refused: bad username or password" << std::endl; break;
            case 5: std::cout << "  -> Connection refused: not authorized" << std::endl; break;
            default: std::cout << "  -> Unknown error" << std::endl; break;
        }
    }
}

void on_publish(struct mosquitto *mosq, void *userdata, int mid) {
    std::cout << "✓ Message published successfully (message ID: " << mid << ")" << std::endl;
}

void on_disconnect(struct mosquitto *mosq, void *userdata, int rc) {
    std::cout << "\n=== DISCONNECTION CALLBACK ===" << std::endl;
    if (rc == 0) {
        std::cout << "✓ Clean disconnection" << std::endl;
    } else {
        std::cout << "✗ Unexpected disconnection (code: " << rc << ")" << std::endl;
    }
    connected = false;
}

void on_log(struct mosquitto *mosq, void *userdata, int level, const char *str) {
    std::cout << "[LOG] " << str << std::endl;
}

bool init_mqtt() {
    std::cout << "=== MQTT INITIALIZATION TEST ===" << std::endl;
    
    // Initialize library
    std::cout << "1. Initializing mosquitto library..." << std::endl;
    mosquitto_lib_init();
    
    // Create client
    std::cout << "2. Creating MQTT client (ID: " << CLIENT_ID << ")..." << std::endl;
    mosq = mosquitto_new(CLIENT_ID.c_str(), true, nullptr);
    if (!mosq) {
        std::cerr << "✗ Failed to create mosquitto client" << std::endl;
        return false;
    }
    std::cout << "✓ MQTT client created" << std::endl;
    
    // Set callbacks
    std::cout << "3. Setting up callbacks..." << std::endl;
    mosquitto_connect_callback_set(mosq, on_connect);
    mosquitto_publish_callback_set(mosq, on_publish);
    mosquitto_disconnect_callback_set(mosq, on_disconnect);
    mosquitto_log_callback_set(mosq, on_log);
    std::cout << "✓ Callbacks set" << std::endl;
    
    // Set credentials
    std::cout << "4. Setting credentials (username: " << USERNAME << ")..." << std::endl;
    int rc = mosquitto_username_pw_set(mosq, USERNAME.c_str(), PASSWORD.c_str());
    if (rc != MOSQ_ERR_SUCCESS) {
        std::cerr << "✗ Failed to set credentials: " << mosquitto_strerror(rc) << std::endl;
        return false;
    }
    std::cout << "✓ Credentials set" << std::endl;
    
    // Set TLS
    std::cout << "5. Setting up TLS/SSL..." << std::endl;
    rc = mosquitto_tls_set(mosq, "/etc/ssl/certs/ca-certificates.crt", nullptr, nullptr, nullptr, nullptr);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::cout << "⚠ Standard CA path failed, trying alternatives..." << std::endl;
        
        // Try alternative paths
        rc = mosquitto_tls_set(mosq, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (rc == MOSQ_ERR_SUCCESS) {
            mosquitto_tls_insecure_set(mosq, true);
            std::cout << "✓ TLS setup successful (insecure mode)" << std::endl;
        } else {
            std::cerr << "✗ All TLS setup attempts failed: " << mosquitto_strerror(rc) << std::endl;
            return false;
        }
    } else {
        std::cout << "✓ TLS setup successful (secure mode)" << std::endl;
    }
    
    return true;
}

bool connect_to_broker() {
    std::cout << "\n=== CONNECTION TEST ===" << std::endl;
    std::cout << "Connecting to: " << MQTT_BROKER << ":" << MQTT_PORT << std::endl;
    
    int rc = mosquitto_connect(mosq, MQTT_BROKER.c_str(), MQTT_PORT, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::cerr << "✗ Connection initiation failed: " << mosquitto_strerror(rc) << std::endl;
        return false;
    }
    
    std::cout << "Connection initiated, waiting for response..." << std::endl;
    
    // Start the network loop
    mosquitto_loop_start(mosq);
    
    // Wait for connection
    int timeout = 10; // 10 second timeout
    while (!connected && timeout > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        timeout--;
        std::cout << "Waiting... (" << timeout << "s remaining)" << std::endl;
    }
    
    return connected;
}

void test_publish() {
    if (!connected) {
        std::cout << "\n✗ Cannot test publish - not connected" << std::endl;
        return;
    }
    
    std::cout << "\n=== PUBLISH TEST ===" << std::endl;
    
    // Test message 1: Simple text
    std::string test_msg1 = "Hello from test script - " + std::to_string(time(nullptr));
    std::cout << "Publishing test message 1..." << std::endl;
    std::cout << "  Topic: " << TEST_TOPIC << std::endl;
    std::cout << "  Payload: " << test_msg1 << std::endl;
    
    int rc = mosquitto_publish(mosq, nullptr, TEST_TOPIC.c_str(), 
                              test_msg1.length(), test_msg1.c_str(), 1, false);
    
    if (rc == MOSQ_ERR_SUCCESS) {
        std::cout << "✓ Publish command sent successfully" << std::endl;
    } else {
        std::cout << "✗ Publish failed: " << mosquitto_strerror(rc) << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    // Test message 2: JSON-like data (simulating NRF data)
    std::string test_msg2 = "{\"classification\":\"person\",\"confidence\":0.85,\"timestamp\":" + std::to_string(time(nullptr)) + "}";
    std::cout << "\nPublishing test message 2 (JSON format)..." << std::endl;
    std::cout << "  Topic: " << TEST_TOPIC << std::endl;
    std::cout << "  Payload: " << test_msg2 << std::endl;
    
    rc = mosquitto_publish(mosq, nullptr, TEST_TOPIC.c_str(), 
                          test_msg2.length(), test_msg2.c_str(), 1, false);
    
    if (rc == MOSQ_ERR_SUCCESS) {
        std::cout << "✓ JSON publish command sent successfully" << std::endl;
    } else {
        std::cout << "✗ JSON publish failed: " << mosquitto_strerror(rc) << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}

void cleanup() {
    std::cout << "\n=== CLEANUP ===" << std::endl;
    
    if (mosq) {
        if (connected) {
            std::cout << "Disconnecting from broker..." << std::endl;
            mosquitto_disconnect(mosq);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        mosquitto_loop_stop(mosq, true);
        mosquitto_destroy(mosq);
        std::cout << "✓ MQTT client destroyed" << std::endl;
    }
    
    mosquitto_lib_cleanup();
    std::cout << "✓ Library cleanup complete" << std::endl;
}

int main() {
    std::cout << "MQTT CONNECTION TEST SCRIPT" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Broker: " << MQTT_BROKER << ":" << MQTT_PORT << std::endl;
    std::cout << "Username: " << USERNAME << std::endl;
    std::cout << "Test Topic: " << TEST_TOPIC << std::endl;
    std::cout << "Client ID: " << CLIENT_ID << std::endl;
    std::cout << std::endl;
    
    // Test 1: Initialize MQTT
    if (!init_mqtt()) {
        std::cout << "\n✗ MQTT initialization failed - exiting" << std::endl;
        cleanup();
        return 1;
    }
    
    // Test 2: Connect to broker
    if (!connect_to_broker()) {
        std::cout << "\n✗ Connection test failed - exiting" << std::endl;
        cleanup();
        return 1;
    }
    
    // Test 3: Publish messages
    test_publish();
    
    // Wait a bit for any final callbacks
    std::cout << "\nWaiting for final callbacks..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    
    // Clean up
    cleanup();
    
    std::cout << "\n=== TEST COMPLETE ===" << std::endl;
    if (connected) {
        std::cout << "✓ All tests passed! Your MQTT parameters are working correctly." << std::endl;
        std::cout << "\nYou can now use these settings in your IoT device:" << std::endl;
        std::cout << "  Broker: " << MQTT_BROKER << std::endl;
        std::cout << "  Port: " << MQTT_PORT << std::endl;
        std::cout << "  Username: " << USERNAME << std::endl;
        std::cout << "  Password: " << PASSWORD << std::endl;
        std::cout << "  Topic pattern: /device/jetson/[your_topic]" << std::endl;
    } else {
        std::cout << "✗ Connection issues detected. Check your parameters and network." << std::endl;
    }
    
    return 0;
}