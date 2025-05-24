#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <sys/ioctl.h>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Skip info messages
        if (severity == Severity::kINFO) return;
        
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            case Severity::kVERBOSE: std::cerr << "VERBOSE: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};


class SerialComm
{
private:
    int serial_fd;
    std::string port;
    int baudrate;
    
public:
    SerialComm(const std::string& port_name) 
        : port(port_name), baudrate(115200), serial_fd(-1)
    {
    }
    
    ~SerialComm()
    {
        if (serial_fd >= 0) {
            close(serial_fd);
        }
    }
    
    bool init()
    {
        // Open serial port
        serial_fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (serial_fd < 0) {
            std::cerr << "Error opening serial port " << port << ": " << strerror(errno) << std::endl;
            return false;
        }
        
        // Configure serial port
        struct termios tty;
        if (tcgetattr(serial_fd, &tty) != 0) {
            std::cerr << "Error getting serial port attributes: " << strerror(errno) << std::endl;
            close(serial_fd);
            serial_fd = -1;
            return false;
        }
        
        // Set baud rate
        cfsetospeed(&tty, getBaudRate(baudrate));
        cfsetispeed(&tty, getBaudRate(baudrate));
        
        // Configure 8N1 (8 data bits, no parity, 1 stop bit)
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
        tty.c_iflag &= ~IGNBRK;         // disable break processing
        tty.c_lflag = 0;                // no signaling chars, no echo, no canonical processing
        tty.c_oflag = 0;                // no remapping, no delays
        tty.c_cc[VMIN]  = 0;            // read doesn't block
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout
        
        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl
        tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls, enable reading
        tty.c_cflag &= ~(PARENB | PARODD);      // shut off odd parity
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;
        
        if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
            std::cerr << "Error setting serial port attributes: " << strerror(errno) << std::endl;
            close(serial_fd);
            serial_fd = -1;
            return false;
        }
        
        std::cout << "Serial port " << port << " opened successfully at " << baudrate << " baud" << std::endl;
        return true;
    }
    
    bool sendData(const std::string& data)
    {
        if (serial_fd < 0) {
            return false;
        }
        
        std::string message = data + "\n"; // Add newline for easier parsing on NRF side
        ssize_t bytes_written = write(serial_fd, message.c_str(), message.length());
        
        if (bytes_written < 0) {
            std::cerr << "Error writing to serial port: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Force write to complete
        tcdrain(serial_fd);
        return true;
    }
    
    bool isConnected() const
    {
        return serial_fd >= 0;
    }

private:
    speed_t getBaudRate(int baud)
    {
        switch (baud) {
            case 9600: return B9600;
            case 19200: return B19200;
            case 38400: return B38400;
            case 57600: return B57600;
            case 115200: return B115200;
            case 230400: return B230400;
            // case 460800: return B460800;
            // case 921600: return B921600;
            default: return B115200;
        }
    }
};

// Destroy TensorRT objects
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
            obj->destroy();
    }
};

// Signal handler for clean shutdown
bool signal_received = false;

void sig_handler(int signo)
{
    if (signo == SIGINT)
    {
        std::cout << "Received SIGINT" << std::endl;
        signal_received = true;
    }
}

// YOLOv8 Inference class using TensorRT
class YOLOv8Inference
{
private:
    Logger logger;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context;
    cudaStream_t stream;
    
    // Input and output buffer pointers
    void* buffers[2]; // Assuming one input, one output
    int inputIndex;
    int outputIndex;
    size_t inputSize;
    size_t outputSize;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    
    int modelWidth;
    int modelHeight;

    // FPS calculation
    float networkFPS;
    std::chrono::steady_clock::time_point lastTime;

    float confidenceThreshold;
    float nmsThreshold;
    int numClasses;

    std::vector<std::string> class_names;

public:
    YOLOv8Inference(const std::string& engineFile) 
        : stream(nullptr), 
          networkFPS(0.0f),
          confidenceThreshold(0.25f), // Default confidence
          nmsThreshold(0.45f),        // Default NMS IOU threshold
          numClasses(80)             // For COCO dataset (YOLOv8n default)
    {
        // Load the engine
        std::cout << "Loading TensorRT engine: " << engineFile << std::endl;
        std::ifstream file(engineFile, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Failed to open engine file: " + engineFile);
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        
        if (!file) {
            throw std::runtime_error("Failed to read engine file");
        }
        
        // Create runtime and deserialize engine
        std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        // Create execution context
        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        // Create CUDA stream
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
        
        // Find input and output binding indices and allocate memory
        inputIndex = -1;
        outputIndex = -1;
        
        for (int i = 0; i < engine->getNbBindings(); i++) {
            if (engine->bindingIsInput(i)) {
                inputIndex = i;
                inputDims = engine->getBindingDimensions(i);
            } else {
                outputIndex = i;
                outputDims = engine->getBindingDimensions(i);
            }
        }
        
        if (inputIndex == -1 || outputIndex == -1) {
            throw std::runtime_error("Could not find input or output binding");
        }

        if (inputDims.nbDims != 4 || inputDims.d[0] != 1 || inputDims.d[1] != 3) {
            throw std::runtime_error("Unexpected input shape. Expected [1, 3, H, W].");
        }
        
        modelHeight = inputDims.d[2];
        modelWidth = inputDims.d[3];
        
        // Calculate sizes and allocate memory
        inputSize = 1;
        for (int i = 0; i < inputDims.nbDims; i++) {
            inputSize *= inputDims.d[i];
        }
        inputSize *= sizeof(float);
        
        outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; i++) {
            outputSize *= outputDims.d[i];
        }
        outputSize *= sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&buffers[inputIndex], inputSize);
        cudaMalloc(&buffers[outputIndex], outputSize);
        
        std::cout << "TensorRT engine loaded successfully" << std::endl;
        std::cout << "Input shape: ";
        for (int i = 0; i < inputDims.nbDims; i++) {
            std::cout << inputDims.d[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Output shape: ";
        for (int i = 0; i < outputDims.nbDims; i++) {
            std::cout << outputDims.d[i] << " ";
        }
        std::cout << std::endl;
        
        // Initialize FPS timer
        lastTime = std::chrono::steady_clock::now();
        
        // Initialize COCO class names
        class_names = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        };
    }
    
    ~YOLOv8Inference()
    {
        // Free allocated resources
        if (buffers[inputIndex]) cudaFree(buffers[inputIndex]);
        if (buffers[outputIndex]) cudaFree(buffers[outputIndex]);
        if (stream) cudaStreamDestroy(stream);
    }

    // Helper for NMS (Non-Maximum Suppression)
    // Detections should be a flat vector of [x1, y1, x2, y2, confidence, class_id]
    std::vector<float> applyNMS(const std::vector<float>& detections, float nmsThreshold) {
        std::vector<float> finalDetections;
        if (detections.empty()) return finalDetections;
    
        // Sort detections by confidence (descending)
        std::vector<int> indices(detections.size() / 6);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return detections[a * 6 + 4] > detections[b * 6 + 4]; // Compare confidence
        });
    
        std::vector<bool> suppressed(detections.size() / 6, false);
    
        for (size_t i = 0; i < indices.size(); ++i) {
            int current_idx = indices[i];
            if (suppressed[current_idx]) continue;

            finalDetections.insert(finalDetections.end(), detections.begin() + current_idx * 6, detections.begin() + (current_idx + 1) * 6);
    
            for (size_t j = i + 1; j < indices.size(); ++j) {
                int next_idx = indices[j];
                if (!suppressed[next_idx]) {
                    float iou = calculateIOU(detections.data() + current_idx * 6, detections.data() + next_idx * 6);
                    if (iou > nmsThreshold) {
                        suppressed[next_idx] = true;
                    }
                }
            }
        }
        return finalDetections;
    }
    
    // Helper function to calculate Intersection over Union (IOU)
    float calculateIOU(const float* box1, const float* box2) {
        float x1_1 = box1[0];
        float y1_1 = box1[1];
        float x2_1 = box1[2];
        float y2_1 = box1[3];
        float x1_2 = box2[0];
        float y1_2 = box2[1];
        float x2_2 = box2[2];
        float y2_2 = box2[3];
    
        float intersection_x1 = std::max(x1_1, x1_2);
        float intersection_y1 = std::max(y1_1, y1_2);
        float intersection_x2 = std::min(x2_1, x2_2);
        float intersection_y2 = std::min(y2_1, y2_2);
    
        float intersection_area = std::max(0.0f, intersection_x2 - intersection_x1) * std::max(0.0f, intersection_y2 - intersection_y1);
    
        float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    
        float union_area = area1 + area2 - intersection_area;
    
        return (union_area > 0) ? intersection_area / union_area : 0.0f;
    }

    std::vector<float> parseYOLOOutput(const std::vector<float>& rawOutput, int frameWidth, int frameHeight) {
        std::vector<float> detections;
        
        // YOLOv8 output: [1, 84, 8400] where 84 = 4 bbox coords + 80 classes
        // Note: YOLOv8 doesn't have explicit objectness score, just class probabilities
        
        if (rawOutput.empty() || outputDims.nbDims < 3) {
            std::cerr << "Error: Invalid output tensor." << std::endl;
            return detections;
        }
    
        const int numPredictions = outputDims.d[2]; // 8400 in [1, 84, 8400]
        const int numElements = outputDims.d[1];    // 84 in [1, 84, 8400]
        
        // Verify we have the expected format
        if (numElements != (numClasses + 4)) {
            std::cerr << "Error: Expected " << (numClasses + 4) << " elements per prediction, got " << numElements << std::endl;
            return detections;
        }
    
        for (int i = 0; i < numPredictions; ++i) {
            // For tensor layout [1, 84, 8400]:
            // Element j for prediction i is at index: j * numPredictions + i
            
            // Extract bounding box coordinates (already normalized 0-1)
            float centerX = rawOutput[0 * numPredictions + i];
            float centerY = rawOutput[1 * numPredictions + i];
            float width = rawOutput[2 * numPredictions + i];
            float height = rawOutput[3 * numPredictions + i];
            
            // Find the class with highest probability (no separate objectness in YOLOv8)
            float maxClassProb = 0.0f;
            int classId = -1;
            for (int c = 0; c < numClasses; ++c) {
                float classProb = rawOutput[(4 + c) * numPredictions + i];
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = c;
                }
            }
            
            // In YOLOv8, the class probability IS the confidence
            float confidence = maxClassProb;
    
            if (confidence > confidenceThreshold) {
                // Convert normalized coordinates to pixel coordinates
                // YOLO outputs are normalized [0,1], so multiply by image dimensions
                float x1 = (centerX - width / 2.0f) * frameWidth;
                float y1 = (centerY - height / 2.0f) * frameHeight;
                float x2 = (centerX + width / 2.0f) * frameWidth;
                float y2 = (centerY + height / 2.0f) * frameHeight;
                
                // Clamp to image boundaries
                x1 = std::max(0.0f, std::min(static_cast<float>(frameWidth), x1));
                y1 = std::max(0.0f, std::min(static_cast<float>(frameHeight), y1));
                x2 = std::max(0.0f, std::min(static_cast<float>(frameWidth), x2));
                y2 = std::max(0.0f, std::min(static_cast<float>(frameHeight), y2));
    
                detections.push_back(x1);
                detections.push_back(y1);
                detections.push_back(x2);
                detections.push_back(y2);
                detections.push_back(confidence);
                detections.push_back(static_cast<float>(classId));
            }
        }
    
        // Apply Non-Maximum Suppression (NMS)
        std::vector<float> finalDetections = applyNMS(detections, nmsThreshold);
        return finalDetections;
    }
    
    // Process a single image
    std::vector<float> processImage(const cv::Mat& image)
    {
        // Start timing for FPS calculation
        auto startTime = std::chrono::steady_clock::now();
        
        // Preprocess image
        cv::Mat resized_rgb;
        cv::resize(image, resized_rgb, cv::Size(modelWidth, modelHeight));
        cv::cvtColor(resized_rgb, resized_rgb, cv::COLOR_BGR2RGB); // Convert to RGB

        // Normalize to [0,1] and convert to float32
        cv::Mat normalized;
        resized_rgb.convertTo(normalized, CV_32F, 1.0f/255.0f);
        
        // Allocate host memory for input
        std::vector<float> inputData(inputSize / sizeof(float));
        
        // Prepare input tensor (NCHW format)
        // inputDims = [1, 3, H, W]
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < modelHeight; ++h) {
                for (int w = 0; w < modelWidth; ++w) {
                    inputData[c * modelHeight * modelWidth + h * modelWidth + w] = 
                        normalized.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        // Copy input data to GPU
        cudaMemcpy(buffers[inputIndex], inputData.data(), inputSize, cudaMemcpyHostToDevice);
        
        // Execute inference
        if (!context->enqueueV2(buffers, stream, nullptr)) {
            throw std::runtime_error("Failed to execute inference");
        }
        
        // Allocate host memory for output
        std::vector<float> outputData(outputSize / sizeof(float));
        
        // Copy output back to host
        cudaMemcpy(outputData.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);
        
        // Synchronize stream
        cudaStreamSynchronize(stream);

        // Parse YOLO output - pass original image dimensions, not model dimensions
        std::vector<float> outputDetections = parseYOLOOutput(outputData, image.cols, image.rows);
        
        // Update FPS calculation
        auto endTime = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        networkFPS = 1000.0f / ms;
        
        return outputDetections;
    }
    
    float GetNetworkFPS() const { return networkFPS; }

    std::string getClassLabel(int classId) const {
        if (classId >= 0 && classId < class_names.size()) {
            return class_names[classId];
        }
        return "unknown";
    }
};

void printUsage()
{
    std::cout << "Usage: yolov8_video_inference <engine_file> <camera_id or video_file>" << std::endl;
    std::cout << "  engine_file: Path to TensorRT engine file (e.g., yolov8n.engine)" << std::endl;
    std::cout << "  camera_id: Camera device ID (e.g., 0 for default camera)" << std::endl;
    std::cout << "  video_file: Path to video file (if using a file instead of camera)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  yolov8_video_inference yolov8n.engine 0" << std::endl;
    std::cout << "  yolov8_video_inference yolov8n.engine /path/to/video.mp4" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printUsage();
        return 1;
    }
    
    // Parse command line
    std::string engineFile = argv[1]; // Now expects a TensorRT engine
    std::string videoSource = argv[2];
    std::string serialPort = argv[3];
    
    // Attach signal handler
    if (signal(SIGINT, sig_handler) == SIG_ERR) {
        std::cerr << "Can't catch SIGINT" << std::endl;
        return 1;
    }

    // Initialize serial communication if port is provided
    std::unique_ptr<SerialComm> serial;
    if (!serialPort.empty()) {
        serial = std::make_unique<SerialComm>(serialPort);
        if (!serial->init()) {
            std::cerr << "Failed to initialize serial communication. Continuing without serial..." << std::endl;
            serial.reset(); // Disable serial communication
        }
    }
    
    // Create video capture
    cv::VideoCapture cap;
    
    // Try to open the video source as a number (camera index)
    try {
        int cameraIndex = std::stoi(videoSource);
        if (!cap.open(cameraIndex)) {
            std::cerr << "Failed to open camera device " << cameraIndex << std::endl;
            return 1;
        }
        std::cout << "Opened camera device " << cameraIndex << std::endl;
    } catch (const std::invalid_argument&) {
        // If not a number, treat as a file path
        if (!cap.open(videoSource)) {
            std::cerr << "Failed to open video file: " << videoSource << std::endl;
            return 1;
        }
        std::cout << "Opened video file: " << videoSource << std::endl;
    }
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Video dimensions: " << width << "x" << height << std::endl;
    
    try {
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Create YOLOv8 inference object
        YOLOv8Inference inference(engineFile); // Pass TensorRT engine path
        
        // Create window for display
        cv::namedWindow("YOLOv8 Inference", cv::WINDOW_NORMAL);
        
        // Processing loop
        cv::Mat frame;
        
        while (!signal_received) {
            // Capture next frame
            if (!cap.read(frame)) {
                std::cout << "End of video stream" << std::endl;
                break;
            }
            
            if (frame.empty()) {
                std::cerr << "Empty frame received" << std::endl;
                continue;
            }
            
            // Run inference on the frame
            std::vector<float> detections = inference.processImage(frame);

            int numDetections = detections.size() / 6;

            std::vector<std::string> detectedClasses;
            
            // Iterate through the detections and draw bounding boxes
            for (int i = 0; i < numDetections; ++i) {
                float x1 = detections[i * 6 + 0];
                float y1 = detections[i * 6 + 1];
                float x2 = detections[i * 6 + 2];
                float y2 = detections[i * 6 + 3];
                float confidence = detections[i * 6 + 4];
                int classId = static_cast<int>(detections[i * 6 + 5]);
                
                // Get the class label
                std::string label = inference.getClassLabel(classId); 

                std::cout << "found " << label << " conf " << confidence << std::endl;
                
                // Add to detected classes list (avoid duplicates)
                if (std::find(detectedClasses.begin(), detectedClasses.end(), label) == detectedClasses.end()) {
                    detectedClasses.push_back(label);
                }

                // Define the color for the bounding box and text
                cv::Scalar color(0, 255, 0); // Green (BGR)

                // Draw the bounding box
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                // Create the text string for label and confidence
                char text[100];
                sprintf(text, "%s: %.2f", label.c_str(), confidence);

                // Calculate the position for the text
                int baseline;
                cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                cv::Point textOrg(x1, y1 - 10);
                if (y1 - 10 < textSize.height) { // Ensure text is visible if box is at top
                    textOrg.y = y1 + textSize.height + 5;
                }

                // Draw the text
                cv::putText(frame, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }

            // Send detection data over serial
            if (serial && serial->isConnected()) {
                if (detectedClasses.empty()) {
                    // Send "NONE" if no objects detected
                    serial->sendData("NONE");
                } else {
                    // Send comma-separated list of detected classes
                    std::string message = "";
                    for (size_t i = 0; i < detectedClasses.size(); ++i) {
                        if (i > 0) {
                            message += ",";
                        } 
                        message += detectedClasses[i];
                    }
                    serial->sendData(message);
                }
            }
            
            char str[100];
            // Display FPS
            sprintf(str, "FPS: %.1f", inference.GetNetworkFPS());
            cv::putText(frame, str, cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Display the frame
            cv::imshow("YOLOv8 Inference", frame);
            
            // Check for keyboard input (press 'q' or ESC to quit)
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            }
        }
        
        // Cleanup
        cap.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Shutdown complete" << std::endl;
    return 0;
}