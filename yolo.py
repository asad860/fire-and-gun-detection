import cv2
import numpy as np
import argparse
import time
from threading import Thread
import uuid
import os
import logging
import smtplib
import ssl
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file",
                    default="videos/fire1.mp4")
parser.add_argument(
    '--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

logging.basicConfig(filename='logs.log', level=logging.DEBUG,
                    format='%(levelname)s:%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
last_mail_time = None


CONFIDENCE = 0.5
ALERT_DIR = 'threats'
sender = "from@example.com"
receiver = "to@example.com"
USERNAME = "user@gmail.com"
PASSWORD = "password"
PORT = 465
SERVER = "smtp.gmail.com"
time_interval = 15

# Load yolo
video_interval = 20  # seconds
video_fps = 2.0


def load_yolo():
    weights_file = "yolov3_custom_last.weights"
    cfg = "yolov3-custom.cfg"
    names = "classes.names"

    net = cv2.dnn.readNet(weights_file, cfg)
    classes = []
    with open(names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f'classes: {classes}')
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1]
                     for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap


def display_blob(blob):
    '''
    Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > CONFIDENCE:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[2]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                img, label+f" confidence: {100*confs[i]:.2f}", (x, y - 5), font, 1, color, 1)

            if label.lower() in ["gun", 'knife', 'rifle']:
                generate_alert(img, label, confs[i])
    return img


def show_image(img):
    img = cv2.resize(img, (600, 400))
    cv2.imshow("Image", img)


def image_log_mail(image_obj, label, conf):
    if not os.path.exists(ALERT_DIR):
        os.mkdir(ALERT_DIR)
    id = str(uuid.uuid4())
    path = os.path.join(ALERT_DIR, id+f'_{label}'+'.jpg')
    cv2.imwrite(path, image_obj)
    logging.warning(f'{label} detected, path: {path}')

    send_mail(path)


def send_mail(filename):
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart

    img_data = open(filename, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Annomly detected.'
    msg['From'] = sender
    msg['To'] = receiver

    text = MIMEText("Report")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(filename))
    msg.attach(image)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SERVER, PORT, context=context) as server:
            server.login(USERNAME, PASSWORD)
            server.sendmail(sender, receiver, msg.as_string())
            logging.info('Annomly detect: Mail sent.')
    except Exception as e:
        logging.error(str(e))


def generate_alert(image_obj, label, conf):
    global last_mail_time
    if last_mail_time == None:
        last_mail_time = datetime.now()
    elif (datetime.now() - last_mail_time).total_seconds() < time_interval:
        print(f'less than {time_interval} seconds:',
              (datetime.now() - last_mail_time).total_seconds())
        return
    print('Sending Mail.')
    last_mail_time = datetime.now()
    thr = Thread(target=image_log_mail, args=[image_obj, label, conf])
    thr.start()


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img = draw_labels(boxes, confs, colors, class_ids, classes, image)
    show_image(img)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('current.avi', fourcc, video_fps, (640,480))
    video_start_time = datetime.now()

    model_fps = 0
    model_start = datetime.now()
    while True:
        ret, frame = cap.read()
        start_frame = datetime.now()

        # ----------------------------------------model-----------------------------
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        frame = draw_labels(boxes, confs, colors, class_ids, classes, frame)
        # -----------------------------------------After Model----------------------

        show_image(frame)
        out.write(frame)

        if ((datetime.now() - video_start_time).seconds > video_interval):
            out.release()
            os.rename('current.avi', 'last.avi')
            video_start_time = datetime.now()
            out = cv2.VideoWriter('current.avi',fourcc, video_fps, (640,480))

        key = cv2.waitKey(1)
        print("time-per-frame: ", (datetime.now() - start_frame).microseconds/10**6, "FPS: ", cap.get(cv2.CAP_PROP_FPS))

        if key == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        img = draw_labels(boxes, confs, colors, class_ids, classes, frame)
        show_image(img)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    webcam = args.webcam
    video_play = args.play_video
    image = args.image
    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        webcam_detect()
    if video_play:
        video_path = args.video_path
        if args.verbose:
            print('Opening '+video_path+" .... ")
        start_video(video_path)
    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening "+image_path+" .... ")
        image_detect(image_path)

    cv2.destroyAllWindows()
