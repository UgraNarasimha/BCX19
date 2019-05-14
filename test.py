import cv2
####################################Usama####
import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
import time

from keras import backend as K
from keras.models import load_model

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval
##########################################################

cap = cv2.VideoCapture(0)

time.sleep(2)
# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
#width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



# We're using // here because in Python // allows for int classical division, 
# because we can't pass a float to the cv2.rectangle function

# Coordinates for Rectangle
#x = width//2
#y = height//2

# Width and height
#w = width//4
#h = height//4


#image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models




# Initiate a session
sess = K.get_session()




while True:
    # Capture frame-by-frame
    
    try:
        ret, frame = cap.read()
        image = frame
        print("Type of image is: ",type(frame))
        width, height = image.size
        image_shape = (height, width)
        boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
        width = np.array(width, dtype=float)
        height = np.array(height, dtype=float)
        image, image_data = preprocess_image(image, model_image_size = (608, 608))
        #resized_image = image.resize(tuple(reversed((608,608)), Image.BICUBIC)
        #image_data = np.array(resized_image, dtype='float32')
        #image_data /= 255.
        #image_data = np.expand_dims(image_data, 0)
        #Run the session
        #out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
        
        
        #Print the results
        print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
        #Produce the colors for the bounding boxs
        colors = generate_colors(class_names)
        #Draw the bounding boxes
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
        #Apply the predicted bounding boxes to the image and save it
        image.save(os.path.join("out", input_image_name), quality=90)
        output_image = scipy.misc.imread(os.path.join("out", input_image_name))
        cv2.imshow('frame', image)
        
    except:
        cap.release()
        cv2.destroyAllWindows()
    # Draw a rectangle on stream
    #cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0,0,255),thickness= 4)

    # Display the resulting frame
   

   # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()