#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
// #include <NvInfer.h> // REMOVE THIS FOR ONNX RUNTIME
// #include <cuda_runtime_api.h> // REMOVE THIS FOR ONNX RUNTIME
#include <onnxruntime_cxx_api.h> // NEW: For ONNX Runtime
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <onnxruntime_c_api.h> 
#include <chrono>

// No need for custom Logger for ONNX Runtime, it has its own logging.
// No need for TRTDestroy struct

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

// YOLOv8 Inference class using ONNX Runtime
class YOLOv8Inference
{
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims; // Output shape will be dynamic based on predictions

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
    YOLOv8Inference(const std::string& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8_Inference_ONNX"),
          session(nullptr),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          networkFPS(0.0f),
          confidenceThreshold(0.25f), // Default confidence
          nmsThreshold(0.45f),        // Default NMS IOU threshold
          numClasses(80)             // For COCO dataset (YOLOv8n default)
    {
        // Setup session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        // Optional: Enable CUDA if available
        // To use CUDA, you need ONNX Runtime built with CUDA support and a CUDA-enabled GPU.
        // If not, it will fall back to CPU.
        // OrtCUDAProviderOptions cuda_options;
        // session_options.AppendExecutionProvider_CUDA(cuda_options);
        
        std::cout << "Loading ONNX model: " << modelPath << std::endl;
        session = Ort::Session(env, modelPath.c_str(), session_options);

        // Get input shape
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        inputDims = tensor_info.GetShape();

        if (inputDims.size() != 4 || inputDims[0] != 1 || inputDims[1] != 3) {
            throw std::runtime_error("Unexpected input shape. Expected [1, 3, H, W]. Got: " + std::to_string(inputDims[0]) + "," + std::to_string(inputDims[1]) + "," + std::to_string(inputDims[2]) + "," + std::to_string(inputDims[3]));
        }
        //modelHeight = inputDims[2];
        //modelWidth = inputDims[3];

        modelHeight = 640;
        modelWidth = 640;

        // Get output shape (this will be dynamic for YOLOv8 detections, but we get first output's shape)
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        outputDims = output_tensor_info.GetShape(); // This will typically be [1, num_features, num_predictions] or [1, num_predictions, num_features]

        std::cout << "ONNX model loaded successfully" << std::endl;
        std::cout << "Input shape: " << inputDims[0] << " " << inputDims[1] << " " << inputDims[2] << " " << inputDims[3] << std::endl;
        std::cout << "Output shape: " << outputDims[0] << " " << outputDims[1] << " " << outputDims[2] << std::endl;


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
    
    // No explicit destructor needed for Ort::Session and Ort::Env as they handle their own memory.

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
        
        if (rawOutput.empty() || outputDims.size() < 3) {
            std::cerr << "Error: Invalid output tensor." << std::endl;
            return detections;
        }
    
        const int numPredictions = outputDims[2]; // 8400 in [1, 84, 8400]
        const int numElements = outputDims[1];    // 84 in [1, 84, 8400]
        
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

    // Parses YOLOv8 output from ONNX Runtime (expected format: [1, 84, num_detections])
    // std::vector<float> parseYOLOOutput(const std::vector<float>& rawOutput, int frameWidth, int frameHeight) {

        
    //     std::vector<float> detections;
        
    //     // YOLOv8 ONNX output is usually [1, num_classes + 4, num_predictions] (e.g., [1, 84, 8400])
    //     // Or sometimes transposed to [1, num_predictions, num_classes + 4]
    //     // Let's assume the common [1, 84, num_predictions] or [1, 80+4, num_predictions] where 80 is numClasses
    //     // Output dimensions: [1, 84, N] where N is number of potential detections (e.g., 8400)
    //     // 84 = 4 (bbox) + 1 (objectness) + 79 (class probabilities) -- adjusted to 80 classes.
    //     // It's 4 bbox (cx,cy,w,h) + objectness + 80 classes = 85.
    //     // So outputDims[1] should be 85 for YOLOv8n COCO
        
    //     if (rawOutput.empty() || outputDims.size() < 3 || outputDims[1] != (numClasses + 4)) {
    //         std::cerr << "Error: Unexpected output tensor shape or size for YOLOv8." << std::endl;
    //         // Optionally print dimensions for debugging
    //         // std::cerr << "OutputDims: ";
    //         // for(long long d : outputDims) std::cerr << d << " ";
    //         // std::cerr << std::endl;
    //         return detections;
    //     }

    //     const int numPredictions = outputDims[2]; // N in [1, 84, N]
    //     const int stride = numClasses + 4; // 84 or 85 if it's 80 classes

    //     // const int numPredictions = rawOutput.size() / 84;

    //     // std::cout << "num_pred" << numPredictions << std::endl;
    
    //     // for (int i = 0; i < numPredictions; ++i) {
    //     //     const float* prediction = rawOutput.data() + i * 84;
    
    //     //     float objectness = prediction[4];
    
    //     //     // Find the class with the highest probability
    //     //     float maxClassProb = 0.0f;
    //     //     int classId = -1;
    //     //     for (int c = 0; c < numClasses; ++c) {
    //     //         if (prediction[5 + c] > maxClassProb) {
    //     //             maxClassProb = prediction[5 + c];
    //     //             classId = c;
    //     //         }
    //     //     }
    
    //     //     float confidence = objectness * maxClassProb;

    //     for (int i = 0; i < numPredictions; ++i) {
    //         // Accessing elements for [1, stride, num_predictions] format
    //         // Data is laid out as: [box1_cx, box1_cy, ..., box2_cx, box2_cy, ..., ]
    //         // So for prediction `i`, element `j` is at `j * num_predictions + i`
            
    //         float objectness = rawOutput[4 * numPredictions + i]; // Confidence score for the object
            
    //         float maxClassProb = 0.0f;
    //         int classId = -1;
    //         for (int c = 0; c < numClasses; ++c) {
    //             float classProb = rawOutput[(5 + c) * numPredictions + i]; // Class probability
    //             if (classProb > maxClassProb) {
    //                 maxClassProb = classProb;
    //                 classId = c;
                    
    //             }
    //         }

    //         float confidence = objectness * maxClassProb;

    //         if (confidence > confidenceThreshold) {
    //             // Extract bounding box coordinates (normalized)
    //             float centerX = rawOutput[0 * numPredictions + i];
    //             float centerY = rawOutput[1 * numPredictions + i];
    //             float width = rawOutput[2 * numPredictions + i];
    //             float height = rawOutput[3 * numPredictions + i];
    
    //             // Convert (cx, cy, w, h) to (x1, y1, x2, y2) and scale to original image size
    //             float x1 = (centerX - width / 2.0f) * frameWidth / modelWidth;
    //             float y1 = (centerY - height / 2.0f) * frameHeight / modelHeight;
    //             float x2 = (centerX + width / 2.0f) * frameWidth / modelWidth;
    //             float y2 = (centerY + height / 2.0f) * frameHeight / modelHeight;
    
    //             detections.push_back(x1);
    //             detections.push_back(y1);
    //             detections.push_back(x2);
    //             detections.push_back(y2);
    //             detections.push_back(confidence);
    //             detections.push_back(static_cast<float>(classId));

    //         }
    //     }
        

    //     // Apply Non-Maximum Suppression (NMS)
    //     std::vector<float> finalDetections = applyNMS(detections, nmsThreshold);

    //     return finalDetections;
    // }
    
    // Process a single image
    // std::vector<float> processImage(const cv::Mat& image)
    // {
    //     // Start timing for FPS calculation
    //     auto startTime = std::chrono::steady_clock::now();
        
    //     // Preprocess image
    //     cv::Mat resized_rgb;
    //     cv::resize(image, resized_rgb, cv::Size(modelWidth, modelHeight));
    //     cv::cvtColor(resized_rgb, resized_rgb, cv::COLOR_BGR2RGB); // Convert to RGB

    //     // Normalize to [0,1] and convert to float32
    //     cv::Mat normalized;
    //     resized_rgb.convertTo(normalized, CV_32F, 1.0f/255.0f);
        
    //     // Prepare input tensor (NCHW format)
    //     // inputDims = [1, 3, H, W]
    //     std::vector<float> inputTensorValues(1 * 3 * modelHeight * modelWidth);
    //     for (int c = 0; c < 3; ++c) {
    //         for (int h = 0; h < modelHeight; ++h) {
    //             for (int w = 0; w < modelWidth; ++w) {
    //                 inputTensorValues[c * modelHeight * modelWidth + h * modelWidth + w] = normalized.at<cv::Vec3f>(h, w)[c];
    //             }
    //         }
    //     }

    //     // Create input tensor for ONNX Runtime
    //     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
    //                                                                inputTensorValues.data(), 
    //                                                                inputTensorValues.size(), 
    //                                                                inputDims.data(), 
    //                                                                inputDims.size());
        
    //     // Try this first, it's often a global function or similar
    //     // Corrected ONNX Runtime API for allocator (using C API function):
    //     // Corrected ONNX Runtime API for allocator (instantiate Ort::AllocatorWithDefaultOptions)
    //     // Create an instance of the AllocatorWithDefaultOptions struct/class.
    //     // This object will handle the default allocator internally.
    //     Ort::AllocatorWithDefaultOptions allocator_with_options;

    //     // Get the raw OrtAllocator* pointer from this object.
    //     // The .get() method is the standard way to retrieve the raw pointer from C++ API wrappers.
    //     OrtAllocator* default_allocator_raw = allocator_with_options;

    //     // Now use this raw allocator pointer for GetInputNameAllocated and GetOutputNameAllocated
    //     Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, default_allocator_raw);
    //     const char* input_names[] = {input_name_ptr.get()};

    //     Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, default_allocator_raw);
    //     const char* output_names[] = {output_name_ptr.get()};


    //     // Execute inference
    //     std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, 
    //                                                     input_names, &input_tensor, 1, 
    //                                                     output_names, 1);
        
    //     // Get output data
    //     float* rawOutputData = ort_outputs[0].GetTensorMutableData<float>();
    //     // The total number of elements in the output tensor
    //     size_t output_tensor_size = ort_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    //     std::vector<float> outputData(rawOutputData, rawOutputData + output_tensor_size);

    //     std::vector<float> outputDetections = parseYOLOOutput(outputData, image.cols, image.rows);
        
    //     // Update FPS calculation
    //     auto endTime = std::chrono::steady_clock::now();
    //     float ms = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    //     networkFPS = 1000.0f / ms;
        
    //     return outputDetections; // Return number of detected objects
    // }
    
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
        
        // Prepare input tensor (NCHW format)
        // inputDims = [1, 3, H, W]
        std::vector<float> inputTensorValues(1 * 3 * modelHeight * modelWidth);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < modelHeight; ++h) {
                for (int w = 0; w < modelWidth; ++w) {
                    inputTensorValues[c * modelHeight * modelWidth + h * modelWidth + w] = 
                        normalized.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // Create input tensor for ONNX Runtime
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                                                                inputTensorValues.data(), 
                                                                inputTensorValues.size(), 
                                                                inputDims.data(), 
                                                                inputDims.size());
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator_with_options;
        OrtAllocator* default_allocator_raw = allocator_with_options;

        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, default_allocator_raw);
        const char* input_names[] = {input_name_ptr.get()};

        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, default_allocator_raw);
        const char* output_names[] = {output_name_ptr.get()};

        // Execute inference
        std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, 
                                                        input_names, &input_tensor, 1, 
                                                        output_names, 1);
        
        // Get output tensor info
        Ort::TensorTypeAndShapeInfo output_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
        outputDims = output_info.GetShape(); // Make sure outputDims is updated!
        
        // Get output data
        float* rawOutputData = ort_outputs[0].GetTensorMutableData<float>();
        size_t output_tensor_size = output_info.GetElementCount();
        std::vector<float> outputData(rawOutputData, rawOutputData + output_tensor_size);

        // Debug output (remove this after confirming it works)
        static bool first_run = true;
        if (first_run) {
            std::cout << "Output tensor shape: ";
            for (int64_t dim : outputDims) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "Total elements: " << output_tensor_size << std::endl;
            first_run = false;
        }

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
    std::cout << "Usage: yolov8_video_inference <onnx_model_file> <camera_id or video_file>" << std::endl;
    std::cout << "  onnx_model_file: Path to ONNX model file (e.g., yolov8n.onnx)" << std::endl;
    std::cout << "  camera_id: Camera device ID (e.g., 0 for default camera)" << std::endl;
    std::cout << "  video_file: Path to video file (if using a file instead of camera)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  yolov8_video_inference yolov8n.onnx 0" << std::endl;
    std::cout << "  yolov8_video_inference yolov8n.onnx /path/to/video.mp4" << std::endl;
}


int main(int argc, char** argv)
{
    if (argc < 3) {
        printUsage();
        return 1;
    }
    
    // Parse command line
    std::string modelFile = argv[1]; // Now expects an ONNX model
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
        // Create YOLOv8 inference object
        YOLOv8Inference inference(modelFile); // Pass ONNX model path
        
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

                std::cout << "found" << label << " conf" << confidence << std::endl;

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
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Shutdown complete" << std::endl;
    return 0;
}