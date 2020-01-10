import cv2

class VehicleDetectionByPlan():

    def __init__(self, model_xml, inference_core, inference_network, executable_network, weights_bin=None, 
    delay=5, confidence_threshold=0.4, thickness=3, color=(0,0,255)):
        self.model_xml = model_xml
        self.inference_core = inference_core
        self.inference_network = inference_network
        self.executable_network = executable_network
        self.weights_bin = weights_bin
        self.infer_requests = {}
        self.delay = delay
        self.confidence_threshold = confidence_threshold
        self.thickness = thickness
        self.color = color
        self.input_blob = next(iter(self.inference_network.inputs))
        self.output_blob = next(iter(self.inference_network.outputs))
        self.bounding_boxes = []

    def obtain_frame(self, cap):
        frame_resize = None
        try:
            ret, frame = cap.read()
            if ret == True:
                frame_resize = cv2.resize(frame,(672,384))
        except Exception as e:
            print(e.args)
            
        return frame_resize

    def apply_network(self, frame, request_id):
        self.infer_requests[request_id] = self.executable_network.start_async(request_id=request_id, inputs={self.input_blob: frame})

    def wait(self):
        for request_id, request in self.infer_requests.items():
            request.wait(self.delay)
    
    def extract_output(self, request_id):
        return self.executable_network.requests[request_id].outputs[self.output_blob]

    def real_time_update(self, frame, xmin, y_min, x_max, y_max):
        cv2.rectangle(frame, (xmin, y_min), (x_max, y_max), self.color, self.thickness)
        return frame

    def get_bounding_boxes(self, output, frame):
        rt_frame = frame.copy()
        for detections in output[0][0]:
            image_id, label, conf, x_min, y_min, x_max, y_max = tuple(detections)
            if conf > self.confidence_threshold:
                rt_frame = self.real_time_update(rt_frame, x_min, y_min, x_max, y_max)
                self.bounding_boxes.append((x_min, y_min, x_max, y_max))

        return rt_frame

