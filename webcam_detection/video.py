import time
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model

from yad2k.models.keras_yolo import yolo_head, yolo_eval
from yad2k.yolo_utils import read_classes, read_anchors, preprocess_webcam_image, draw_boxes, \
    generate_colors
from socket import * # Import necessary modules

HOST = '100.100.198.163'    # Server(Raspberry Pi) IP address
PORT = 21567
BUFSIZ = 1024             # buffer size
ADDR = (HOST, PORT)


tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)                    # Connect with the server



	
	
	


stream = cv2.VideoCapture(0)

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (480., 640.)

yolo_model = load_model("model_data/yolo.h5")
print(yolo_model.summary())
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes, box_xy, box_wh = yolo_eval(yolo_outputs, image_shape)


def predict(sess, frame):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your Tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    image, image_data = preprocess_webcam_image(frame, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                       K.learning_phase(): 0})
    # Print predictions info
    print('Found {} boxes'.format(len(out_boxes)))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    cars = np.count_nonzero(out_classes == 2)
    return np.array(image), cars

    #return np.array(image)


sess = K.get_session()

(grabbed, frame) = stream.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]
#print fwidth , fheight
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (fwidth,fheight))


#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('output', fourcc, 24.0, ())

while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break

    # Run detection
    start = time.time()
    output_image, cars = predict(sess, frame)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    print("The number of cars: ",cars)

    if cars >= 3:
        tcpCliSock.send(b"1")
        print("green")
    else:
        tcpCliSock.send(b"0")
        print("red")
    

    # Display the resulting frame
    cv2.imshow('', output_image)
    out.write(output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
stream.release()
out.release()
cv2.destroyAllWindows()
tcpCliSock.close()
