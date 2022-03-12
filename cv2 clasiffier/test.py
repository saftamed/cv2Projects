import cv2

image_path = "cc.jpg"
window_name = f"Detected Objects in {image_path}"
original_image = cv2.imread(image_path)

# Convert the image to grayscale for easier computation
image_grey = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

cascade_classifier = cv2.CascadeClassifier(
    f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    # Get frames
    ret, frame = cap.read()
    image_grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    detected_objects = cascade_classifier.detectMultiScale(image_grey, minSize=(50, 50))

    # Draw rectangles on the detected objects
    if len(detected_objects) != 0:
        for (x, y, width, height) in detected_objects:
            cv2.rectangle(frame, (x, y),
                          (x + height, y + width),
                          (0, 255, 0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, frame)
    cv2.resizeWindow(window_name, 400, 400)
    key = cv2.waitKey(1)
    if key == 116:
        break
cv2.destroyAllWindows()


"""

haarcascade_eye.xml
haarcascade_eye_tree_eyeglasses.xml
haarcascade_frontalcatface.xml
haarcascade_frontalcatface_extended.xml
haarcascade_frontalface_alt.xml
haarcascade_frontalface_alt2.xml
haarcascade_frontalface_alt_tree.xml
haarcascade_frontalface_default.xml
haarcascade_fullbody.xml
haarcascade_lefteye_2splits.xml
haarcascade_licence_plate_rus_16stages.xml
haarcascade_lowerbody.xml
haarcascade_profileface.xml
haarcascade_righteye_2splits.xml
haarcascade_russian_plate_number.xml
haarcascade_smile.xml
haarcascade_upperbody.xml
"""