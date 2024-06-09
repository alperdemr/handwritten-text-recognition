import cv2 as cv
import numpy as np
import tensorflow as tf
import gradio as gr


def process_image(image_path,model_path="handwriting.model"):
    # load the handwriting OCR model
    print("[INFO] loading handwriting OCR model...")
    model = tf.keras.models.load_model(model_path)

    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv.Canny(blurred, 30, 150)
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv.boundingRect(x)[0])

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv.threshold(roi, 0, 255,
                cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = cv.resize(thresh, (32, 32))

            # otherwise, resize along the height
            else:
                thresh = cv.resize(thresh, (32, 32))

            # re-grab the image dimensions (now that its been resized)
            # and
            # then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv.copyMakeBorder(thresh, dY, dY,
                dX, dX, cv.BORDER_CONSTANT,
                value=(0, 0, 0))
            padded = cv.resize(padded, (32, 32))
            padded = padded.astype("float32") / 255.0 
            padded = np.expand_dims(padded, axis=-1)
            chars.append((padded, (x, y, w, h)))


    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    preds = model.predict(chars)
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest probability, then
        # extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]

        # draw the prediction on the image
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, label, (x - 10, y - 10),
            cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    return image

def ocr_handwriting(image):
    image_path = "temp_image.jpg"
    cv.imwrite(image_path, image)

    processed_image = process_image(image_path)

    processed_image = cv.cvtColor(processed_image, cv.COLOR_BGR2RGB)
    return processed_image

iface = gr.Interface(fn=ocr_handwriting, inputs="image", outputs="image")
iface.launch()