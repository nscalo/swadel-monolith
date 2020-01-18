from handle_models import handle_output, preprocessing
from inference import Network
import argparse
import cv2
import sys
import numpy as np
from app.service.SpatialInformationService import SpatialInformationService
sys.path.append('/home/aswin/anaconda3/lib')

def get_args():
    '''
    Gets the arguments from the command line.
    '''

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_args()
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    bounding_boxes = []

    # Read the input image
    image = cv2.imread(args.i)

    ### TODO: Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()

    for box in output[0][0]:
        for detections in output[0][0]:
            image_id, label, conf, x_min, y_min, x_max, y_max = tuple(detections)
            if conf > 0.5:
                bounding_boxes.append((x_min, y_min, x_max, y_max))
                bounding_box = (x_min, y_min, x_max, y_max)
                frame = image[x_min:x_max,y_min:y_max]
                break

    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0,255,0), thickness=3)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
