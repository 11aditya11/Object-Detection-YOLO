import io
import os
from google.cloud import vision_v1p3beta1 as vision
from datetime import datetime
import cv2

# Setup google authen client key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'OCR-task-8c6b6f4f9e26.json'

# Source path content all images
# SOURCE_PATH = "C:/Users/eshan/Desktop/ALPR"


def recognize_license_plate(img_path):

    start_time = datetime.now()

    # Read image with opencv
    img = img_path

    # Get image size
    height, width = img.shape[:2]

    # Scale image
    #img = cv2.resize(img, (600, int((height * 600) / width)))

    # Show the origin image
    #cv2.imshow('Origin image', img)

    # Save the image to temp file
    cv2.imwrite( "output.jpg", img)

    # Create new img path for google vision
    img_path =  "output.jpg"

    # Create google vision client
    client = vision.ImageAnnotatorClient()

    # Read image file
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    # Recognize text
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        if len(text.description) :
            license_plate = text.description
            print(text.description)
            vertices = [(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices]

            # Put text license plate number to image
            #cv2.putText(img, license_plate, (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            print(vertices)
            # Draw rectangle around license plate
            #cv2.rectangle(img, (vertices[0][0]-10, vertices[0][1]-10), (vertices[2][0]+10, vertices[2][1]+10), (0, 255, 0), 3)
            print('Total time: {}'.format(datetime.now() - start_time))
            #cv2.imshow('Recognize & Draw', img)
            #cv2.imwrite("output.jpg",img)
            cv2.waitKey(0)
            return (vertices[0][0]-10, vertices[0][1]-10, vertices[2][0]+10, vertices[2][1]+10, license_plate)



