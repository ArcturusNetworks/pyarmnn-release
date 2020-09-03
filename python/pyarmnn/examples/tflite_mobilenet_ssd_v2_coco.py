import numpy as np
import pyarmnn as ann
import example_utils as eu
import time

labels_filename = '/usr/share/OpenCV/samples/data/dnn/object_detection_classes_coco.txt'
model_filename = '/usr/share/deepvision/data/models/objdet/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tflite'
image_filename = '/usr/share/deepvision/data/images/bicycle_race.jpg'

# Create a network from a model file
net_id, graph_id, parser, runtime = eu.create_tflite_network(model_filename)

# Load input information from the model and create input tensors
input_names = parser.GetSubgraphInputTensorNames(graph_id)
input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
input_width = input_binding_info[1].GetShape()[1]
input_height = input_binding_info[1].GetShape()[2]

# Load output information from the model and create output tensors
output_names = parser.GetSubgraphOutputTensorNames(graph_id)
output_bind_info = []
for output_name in output_names:
        output_bind_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))

output_tensors = ann.make_output_tensors(output_bind_info)

# Load labels file
labels = eu.load_labels(labels_filename)

# Load images and resize to expected size
image_names = [image_filename]
images = eu.load_images(image_names, input_width, input_height)

for idx, im in enumerate(images):
        for i in range(2):
                # Create input tensors
                input_tensors = ann.make_input_tensors([input_binding_info], [im])

                # Run inference
                print("Running inference on '{0}' ...".format(image_names[idx]))
                a = time.time()
                runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
                b = time.time()
                # Process output
                results = ann.workload_tensors_to_ndarray(output_tensors)
                bboxes = results[0][0]
                classes = results[1][0]
                scores = results[2][0]
                numDets = results[3][0]
                for label,conf in zip(results[1],results[2]):
                        print("{} - {}%".format(labels[int(label)], int(conf*100)))
                print("Inference time: {} ms".format((b-a)*1000.0))
