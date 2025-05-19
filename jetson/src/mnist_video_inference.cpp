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

// MNIST Inference class
class MNISTInference
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
    
    // FPS calculation
    float networkFPS;
    std::chrono::steady_clock::time_point lastTime;

public:
    MNISTInference(const std::string& engineFile) : stream(nullptr), networkFPS(0)
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
    }
    
    ~MNISTInference()
    {
        // Free allocated resources
        if (buffers[inputIndex]) cudaFree(buffers[inputIndex]);
        if (buffers[outputIndex]) cudaFree(buffers[outputIndex]);
        if (stream) cudaStreamDestroy(stream);
    }

    std::vector<float> applyNMS(const std::vector<float>& detections, float nmsThreshold) {
        std::vector<float> finalDetections;
        if (detections.empty()) return finalDetections;
    
        // Sort detections by confidence (descending)
        std::vector<int> indices(detections.size() / 6);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return detections[a * 6 + 4] > detections[b * 6 + 4];
        });
    
        std::vector<bool> suppressed(detections.size() / 6, false);
    
        for (int i = 0; i < indices.size(); ++i) {
            if (suppressed[indices[i]]) continue;
            finalDetections.insert(finalDetections.end(), detections.begin() + indices[i] * 6, detections.begin() + (indices[i] + 1) * 6);
    
            for (int j = i + 1; j < indices.size(); ++j) {
                if (!suppressed[indices[j]]) {
                    float iou = calculateIOU(detections.data() + indices[i] * 6, detections.data() + indices[j] * 6);
                    if (iou > nmsThreshold) {
                        suppressed[indices[j]] = true;
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

    std::vector<float> MNISTInference::parseYOLOOutput(const std::vector<float>& rawOutput, int modelWidth, int modelHeight, int frameWidth, int frameHeight) {
        std::vector<float> detections;
        float confidenceThreshold = 0.5f; // Adjust as needed
        float nmsThreshold = 0.5f;        // Adjust as needed
        int numClasses = 80;             // For COCO
    
        // Assuming the output tensor has a shape like [1, num_predictions, 85]
        const int numPredictions = rawOutput.size() / 85;
    
        for (int i = 0; i < numPredictions; ++i) {
            const float* prediction = rawOutput.data() + i * 85;
    
            float objectness = prediction[4];
    
            // Find the class with the highest probability
            float maxClassProb = 0.0f;
            int classId = -1;
            for (int c = 0; c < numClasses; ++c) {
                if (prediction[5 + c] > maxClassProb) {
                    maxClassProb = prediction[5 + c];
                    classId = c;
                }
            }
    
            float confidence = objectness * maxClassProb;
    
            if (confidence > confidenceThreshold) {
                // Extract bounding box coordinates (normalized)
                float centerX = prediction[0];
                float centerY = prediction[1];
                float width = prediction[2];
                float height = prediction[3];
    
                // Scale bounding box to the original image size
                float x1 = (centerX - width / 2.0f) * frameWidth;
                float y1 = (centerY - height / 2.0f) * frameHeight;
                float x2 = (centerX + width / 2.0f) * frameWidth;
                float y2 = (centerY + height / 2.0f) * frameHeight;
    
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
    int processImage(const cv::Mat& image, std::vector<float>& outputDetections)
    {
        // Start timing for FPS calculation
        auto startTime = std::chrono::steady_clock::now();
        
        // Get dimensions
        const int batchSize = inputDims.d[0];
        const int channels = inputDims.d[1];
        const int modelHeight = inputDims.d[2];
        const int modelWidth = inputDims.d[3];
        
        // Resize and preprocess image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(modelWidth, modelHeight));
        
        // Convert to grayscale if needed
        // cv::Mat gray;
        // if (channels == 1 && resized.channels() == 3) {
        //     cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        // } else if (channels == 3 && resized.channels() == 1) {
        //     cv::cvtColor(resized, gray, cv::COLOR_GRAY2BGR);
        // } else {
        //     gray = resized;
        // }

        cv::Mat converted;
        cv::cvtColor(resized, converted, cv::COLOR_BGR2RGB);
        
        // Normalize to [0,1]
        cv::Mat normalized;
        // gray.convertTo(normalized, CV_32F, 1.0f/255.0f);
        converted.convertTo(normalized, CV_32F, 1.0f/255.0f);
        
        // Allocate host memory for input
        std::vector<float> inputData(inputSize / sizeof(float));
        
        // Copy data to input buffer (assuming NCHW format)
        if (channels == 3) {
            // For color, copy each channel
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < modelHeight; h++) {
                    for (int w = 0; w < modelWidth; w++) {
                        inputData[(c * modelHeight * modelWidth) + (h * modelWidth) + w] = 
                            normalized.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
        } else {
            // For grayscale, just copy the data
            for (int h = 0; h < modelHeight; h++) {
                for (int w = 0; w < modelWidth; w++) {
                    inputData[h * modelWidth + w] = normalized.at<float>(h, w);
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

        outputDetections = parseYOLOOutput(outputData, modelWidth, modelHeight, frame.cols, frame.rows);
        
        // // Find the class with highest confidence (MNIST has 10 classes, 0-9)
        // int numClasses = outputDims.d[1];  // Assuming output shape is [batch_size, num_classes]
        // auto maxElement = std::max_element(outputData.begin(), outputData.begin() + numClasses);
        // int classId = std::distance(outputData.begin(), maxElement);
        
        // Set confidence value if requested
        // if (confidence != nullptr) {
        //     *confidence = *maxElement;
        // }
        
        // Update FPS calculation
        auto endTime = std::chrono::steady_clock::now();
        
        // Calculate time in ms
        float ms = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        
        // Update running average of network time
        networkFPS = 1000.0f / ms;
        
        return outputDetections.size();
    }
    
    float GetNetworkFPS() const { return networkFPS; }
};

void printUsage()
{
    std::cout << "Usage: mnist_video_inference <engine_file> <camera_id or video_file>" << std::endl;
    std::cout << "  engine_file: Path to TensorRT engine file" << std::endl;
    std::cout << "  camera_id: Camera device ID (e.g., 0 for default camera)" << std::endl;
    std::cout << "  video_file: Path to video file (if using a file instead of camera)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  mnist_video_inference alexnet_mnist.engine 0" << std::endl;
    std::cout << "  mnist_video_inference alexnet_mnist.engine /path/to/video.mp4" << std::endl;
}

std::string getClassLabel(int classId) {
    switch (classId) {
        case 0: return "person";
        case 1: return "bicycle";
        case 2: return "car";
        case 3: return "motorcycle";
        case 4: return "airplane";
        case 5: return "bus";
        case 6: return "train";
        case 7: return "truck";
        case 8: return "boat";
        case 9: return "traffic light";
        case 10: return "fire hydrant";
        case 11: return "stop sign";
        case 12: return "parking meter";
        case 13: return "bench";
        case 14: return "bird";
        case 15: return "cat";
        case 16: return "dog";
        case 17: return "horse";
        case 18: return "sheep";
        case 19: return "cow";
        case 20: return "elephant";
        case 21: return "bear";
        case 22: return "zebra";
        case 23: return "giraffe";
        case 24: return "backpack";
        case 25: return "umbrella";
        case 26: return "handbag";
        case 27: return "tie";
        case 28: return "suitcase";
        case 29: return "frisbee";
        case 30: return "skis";
        case 31: return "snowboard";
        case 32: return "sports ball";
        case 33: return "kite";
        case 34: return "baseball bat";
        case 35: return "baseball glove";
        case 36: return "skateboard";
        case 37: return "surfboard";
        case 38: return "tennis racket";
        case 39: return "bottle";
        case 40: return "wine glass";
        case 41: return "cup";
        case 42: return "fork";
        case 43: return "knife";
        case 44: return "spoon";
        case 45: return "bowl";
        case 46: return "banana";
        case 47: return "apple";
        case 48: return "sandwich";
        case 49: return "orange";
        case 50: return "broccoli";
        case 51: return "carrot";
        case 52: return "hot dog";
        case 53: return "pizza";
        case 54: return "donut";
        case 55: return "cake";
        case 56: return "chair";
        case 57: return "couch"; // Corrected from 'sofa'
        case 58: return "potted plant"; // Corrected spacing
        case 59: return "bed";
        case 60: return "dining table"; // Corrected spacing
        case 61: return "toilet";
        case 62: return "tv"; // Shortened from 'tvmonitor'
        case 63: return "laptop";
        case 64: return "mouse";
        case 65: return "remote";
        case 66: return "keyboard";
        case 67: return "cell phone"; // Corrected spacing
        case 68: return "microwave";
        case 69: return "oven";
        case 70: return "toaster";
        case 71: return "sink";
        case 72: return "refrigerator";
        case 73: return "book";
        case 74: return "clock";
        case 75: return "vase";
        case 76: return "scissors";
        case 77: return "teddy bear"; // Corrected spacing
        case 78: return "hair drier"; // Corrected spacing
        case 79: return "toothbrush";
        default: return "unknown";
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printUsage();
        return 1;
    }
    
    // Parse command line
    std::string engineFile = argv[1];
    std::string videoSource = argv[2];
    
    // Attach signal handler
    if (signal(SIGINT, sig_handler) == SIG_ERR) {
        std::cerr << "Can't catch SIGINT" << std::endl;
        return 1;
    }
    
    // Create video capture
    cv::VideoCapture cap;
    
    // Try to open the video source as a number (camera index)
    try {
        // int cameraIndex = std::stoi(videoSource);
        int cameraIndex = 0;
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
        
        // Create MNIST inference object
        MNISTInference inference(engineFile);
        
        // Create window for display
        cv::namedWindow("MNIST Inference", cv::WINDOW_NORMAL);
        
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
            std::vector<float> detections;
            inference.processImage(frame, detections);
            // int digit = inference.processImage(frame, &confidence);
            
            // Convert confidence to percentage
            // confidence *= 100.0f;
            
            // Draw the recognized digit and confidence
            // char str[100];
            // sprintf(str, "Digit: %d (%.2f%%)", digit, confidence);
            // cv::putText(frame, str, cv::Point(10, 30), 
            //             cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            // Iterate through the detections and draw bounding boxes
            for (size_t i = 0; i < detections.size(); i += 6) { // Assuming each detection has 6 elements
                float x1 = detections[i];
                float y1 = detections[i + 1];
                float x2 = detections[i + 2];
                float y2 = detections[i + 3];
                float confidence = detections[i + 4];
                int classId = static_cast<int>(detections[i + 5]);

                // Get the class label based on the classId
                std::string label = getClassLabel(classId); // You'll need to implement this function

                // Define the color for the bounding box and text
                cv::Scalar color(0, 255, 0); // Green

                // Draw the bounding box
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                // Create the text string for label and confidence
                char text[100];
                sprintf(text, "%s: %.2f", label.c_str(), confidence);

                // Calculate the position for the text
                int baseline;
                cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                cv::Point textOrg(x1, y1 - 10);

                // Draw the text
                cv::putText(frame, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }
            
            // Display FPS
            sprintf(str, "FPS: %.1f", inference.GetNetworkFPS());
            cv::putText(frame, str, cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Display the frame
            cv::imshow("MNIST Inference", frame);
            
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