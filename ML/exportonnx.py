"""
Standalone script to convert a trained Matterport Mask R-CNN Keras .h5 model to ONNX format.

Usage:
    python3 export_onnx_model.py \
        --model_path=/path/to/your/trained_model.h5 \
        --output_onnx_path=/path/to/save/your_model.onnx \
        --dataset_dir=../data \
        --class_map=./taco_config/map_10.csv

    python3 exportonnx.py --model_path=mask_rcnn_coco.h5 --output_onnx_path=./model.onnx --dataset_dir=../data --class_map=./taco_config/map_10.csv

Ensure you have a Python 3.6 environment with TensorFlow 1.15, Keras 2.2.5,
h5py 2.10.0, numpy 1.18, and keras2onnx installed.
The project's 'dataset.py', 'model.py', 'config.py', and 'utils.py' files
must be accessible (e.g., in the same directory).
"""

import os
import argparse
import tensorflow as tf
import keras
from keras.models import Model
import keras2onnx
import csv

# Import necessary components from the original project
# Make sure these files (dataset.py, model.py, config.py, utils.py)
# are in the same directory as this script, or accessible via PYTHONPATH.
from dataset import Taco
from model import MaskRCNN
from config import Config
import utils # utils is imported by MaskRCNN/Config and possibly needed by download_trained_weights


def export_keras_to_onnx(model_h5_path, output_onnx_path, dataset_dir, class_map_path):
    """
    Loads a trained Keras Mask R-CNN model and converts it to ONNX format.

    Args:
        model_h5_path (str): Path to the trained Keras .h5 model weights file.
        output_onnx_path (str): Path where the ONNX model will be saved.
        dataset_dir (str): Directory of the dataset (needed to infer num_classes for Config).
        class_map_path (str): Path to the CSV file defining target classes (needed for Config).
    """
    print(f"Starting ONNX export for model: {model_h5_path}")

    # --- 1. Determine Number of Classes for Model Configuration ---
    # We need to correctly instantiate the MaskRCNN Config which requires NUM_CLASSES.
    # This is typically derived from the dataset the model was trained on.
    # A lightweight way to get this without loading full data is to use a dummy dataset load.
    class_map = {}
    with open(class_map_path) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}

    dummy_dataset = Taco()
    # Assuming 'round 0' and 'test' split is sufficient to derive class count
    dummy_dataset.load_taco(dataset_dir, 0, "test", class_map=class_map, auto_download=None)
    dummy_dataset.prepare()
    nr_classes = dummy_dataset.num_classes
    print(f"Inferred number of classes: {nr_classes}")

    # --- 2. Configure Mask R-CNN for Inference ---
    # Create a dummy TestConfig. Important parameters are GPU_COUNT=1, IMAGES_PER_GPU=1, NUM_CLASSES.
    class InferenceConfig(Config):
        NAME = "taco_inference"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = nr_classes # Must match the model's trained classes
        # Set IMAGE_MAX_DIM to a standard value that the model was likely trained with,
        # e.g., 1024 as per original Matterport Mask R-CNN default or your training config.
        # This is important for the input shape of the ONNX model.
        IMAGE_MIN_DIM = 800
        IMAGE_MAX_DIM = 1024
        USE_MINI_MASK = True # If your model uses mini masks, keep this.
        MINI_MASK_SHAPE = (56, 56) # Default for Matterport Mask R-CNN
        # Other settings can be defaults for inference
        DETECTION_MIN_CONFIDENCE = 0.7 # Or whatever confidence threshold makes sense for your deployment
        DETECTION_NMS_THRESHOLD = 0.3


    config = InferenceConfig()
    config.display()

    # --- 3. Create Mask R-CNN Model in Inference Mode ---
    print("Creating Mask R-CNN model in inference mode...")
    model_instance = MaskRCNN(mode="inference", config=config, model_dir=".") # model_dir not strictly needed for export

    # --- 4. Load Trained Weights ---
    print(f"Loading trained weights from {model_h5_path}...")
    # The `model_instance.load_weights` call internally builds the Keras model graph
    # and loads the weights into it.
    
    # Handle COCO pre-trained weights if the path is 'coco' related
    if model_h5_path.lower() == "coco": # This handles the case if user provides "coco" string
        # Assuming COCO_MODEL_PATH is defined somewhere, or download it.
        # For a standalone script, let's hardcode or pass the path.
        # Default Matterport COCO weights path (if not already downloaded)
        coco_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_rcnn_coco.h5")
        if not os.path.exists(coco_weights_path):
            print(f"Downloading COCO weights to {coco_weights_path}...")
            # utils.download_trained_weights expects a path to download to.
            # Make sure utils.download_trained_weights is correctly imported and functional.
            utils.download_trained_weights(coco_weights_path)
        model_h5_path = coco_weights_path

        # Exclude the last layers for COCO weights as they differ in class count
        model_instance.load_weights(model_h5_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        # Load custom trained weights
        model_instance.load_weights(model_h5_path, by_name=True)

    # --- 5. Prepare Keras Model for ONNX Conversion ---
    # Crucial for TensorFlow 1.x based Keras models.
    # Clears any previous Keras session and sets the backend to use TF 1.x session.
    keras.backend.clear_session()
    # It's better to explicitly grab the session from the loaded model instance.
    # Matterport's Mask R-CNN in TF 1.x context builds its graph within a session.
    # The MaskRCNN object often stores the session in `model.sess`.
    try:
        session = model_instance.sess # For TF 1.x MaskRCNN
        keras.backend.set_session(session)
    except AttributeError:
        # If model_instance.sess doesn't exist (e.g., if TF 2.x was used and `model_instance` is a Keras Model directly),
        # then this step is not needed or handled differently by Keras itself.
        # But for TF 1.x, this is typically how the session is managed.
        print("Warning: model_instance.sess not found. Assuming Keras backend session is managed automatically.")
        pass

    # The actual Keras model object is usually found in `model_instance.keras_model`
    keras_model_to_export = model_instance.keras_model

    if not isinstance(keras_model_to_export, Model):
        raise TypeError("The MaskRCNN model object does not expose a Keras Model instance correctly.")
    print("Keras model loaded successfully.")

    # --- 6. Convert Keras Model to ONNX ---
    print("Converting Keras model to ONNX format...")
    # Define input shape for ONNX conversion.
    # Mask R-CNN model's input is typically `(None, IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)`
    # where None is batch size. For ONNX export, it's often best to specify a fixed batch size.
    # Using 1 for inference.
    input_shape = [(None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[2])] # List of tuples for inputs
    
    # You might want to explicitly name inputs/outputs for clarity in ONNX Runtime.
    # Inspect `keras_model_to_export.input_names` and `keras_model_to_export.output_names`
    # to get exact names if needed.
    # For a general export, default names are often fine.
    
    onnx_model = keras2onnx.convert_keras(
        keras_model_to_export,
        name=os.path.basename(output_onnx_path).split('.')[0],
        target_opset=10, # Opset 10 is generally a good balance for compatibility.
        # input_names=keras_model_to_export.input_names, # Use existing names if known
        # output_names=keras_model_to_export.output_names # Use existing names if known
    )
    print("Keras model converted to ONNX.")

    # --- 7. Save ONNX Model ---
    print(f"Saving ONNX model to {output_onnx_path}...")
    keras2onnx.save_model(onnx_model, output_onnx_path)
    print(f"ONNX model successfully saved to: {output_onnx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Mask R-CNN Keras .h5 to ONNX.')
    parser.add_argument('--model_path', required=True,
                        help='Path to the trained Keras .h5 model weights file (or "coco" for COCO weights).')
    parser.add_argument('--output_onnx_path', required=True,
                        help='Path to save the output ONNX model.')
    parser.add_argument('--dataset_dir', required=True,
                        help='Directory of the dataset (e.g., ../data). Needed to infer NUM_CLASSES.')
    parser.add_argument('--class_map', required=True,
                        help='Path to the CSV file defining target classes (e.g., ./taco_config/map_10.csv). Needed for NUM_CLASSES.')

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_onnx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        export_keras_to_onnx(
            args.model_path,
            args.output_onnx_path,
            args.dataset_dir,
            args.class_map
        )
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")
        # Optionally, add more specific error handling for keras2onnx related issues.