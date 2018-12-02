from __future__ import division
import cv2
import dlib
import time
import sys
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import cg
import math
from linebot.exceptions import LineBotApiError
from flask import Flask, request, abort
from imgurpython import ImgurClient
from imutils import face_utils, rotate_bound
from PIL import Image, ImageDraw, ImageFont

from flask import Flask, request, abort
from imgurpython import ImgurClient

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

import tempfile, os
from config import client_id, client_secret, album_id, access_token, refresh_token, line_channel_access_token, \
    line_channel_secret

app = Flask(__name__)

line_bot_api = LineBotApi(line_channel_access_token)
handler = WebhookHandler(line_channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "./NotoSansCJK-Black.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def detectFaceDlibHog(detector, frame, inHeight=1150, inWidth=0):
    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibHogSmall, 0)
    print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:
        cvRect = [int(faceRect.left() * scaleWidth), int(faceRect.top() * scaleHeight),
                  int(faceRect.right() * scaleWidth), int(faceRect.bottom() * scaleHeight)]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      int(round(frameHeight / 150)), 4)
    return frameDlibHog, bboxes, faceRects


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    h, w, _ = overlay_color.shape

    roi = bg_img[y:y + h, x:x + w]

    # Convert uint8 to float
    foreground = overlay_color.astype(float)
    background = bg_img.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = cv2.merge((a, a, a))
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    return cv2.add(foreground, background)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def adjust_sprite2head(sprite, head_width, head_ypos, ontop = True):
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0*head_width/w_sprite
    sprite = cv2.resize(sprite, (0,0), fx=factor, fy=factor) # adjust to have the same width as head
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig =  head_ypos-h_sprite if ontop else head_ypos # adjust the position of sprite to end where the head begins
    if (y_orig < 0): 
            sprite = sprite[abs(y_orig)::,:,:] #in that case, we cut the sprite
            y_orig = 0 #the sprite then begins at the top of the image
    return (sprite, y_orig)

def draw_sprite(frame, sprite, x_offset, y_offset):
    (h,w) = (sprite.shape[0], sprite.shape[1])
    (imgH,imgW) = (frame.shape[0], frame.shape[1])

    if y_offset+h >= imgH: #if sprite gets out of image in the bottom
        sprite = sprite[0:imgH-y_offset,:,:]

    if x_offset+w >= imgW: #if sprite gets out of image to the right
        sprite = sprite[:,0:imgW-x_offset,:]

    if x_offset < 0: #if sprite gets out of image to the left
        sprite = sprite[:,abs(x_offset)::,:]
        w = sprite.shape[1]
        x_offset = 0

    #for each RGB chanel
    for c in range(3):
            #chanel 4 is alpha: 255 is not transpartne, 0 is transparent background
            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] =  \
            sprite[:,:,c] * (sprite[:,:,3]/255.0) +  frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - sprite[:,:,3]/255.0)
    return frame

#points are tuples in the form (x,y)
# returns angle between points in degrees
def calculate_inclination(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    return incl

####################################################################

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    # print("body:",body)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'ok'


@handler.add(MessageEvent, message=(ImageMessage, TextMessage))
def handle_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
        message_content = line_bot_api.get_message_content(event.message.id)
        with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
            for chunk in message_content.iter_content():
                tf.write(chunk)
            tempfile_path = tf.name

        dist_path = tempfile_path + '.' + ext
        dist_name = os.path.basename(dist_path)
        os.rename(tempfile_path, dist_path)

        hogFaceDetector = dlib.get_frontal_face_detector()

        model = "./shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(model)
        t_face = cv2.imread(dist_path)

        # t_face, bboxes, faceRects = detectFaceDlibHog(hogFaceDetector,t_face)
        t_gray = cv2.cvtColor(t_face, cv2.COLOR_BGR2GRAY)
        faces = hogFaceDetector(t_gray, 0)

        numFaces = len(faces)

        if numFaces == 0:
            overlay_papa = cv2.imread("./tp_papa.png", -1)
            overlay_out = overlay_papa
            t_w = t_face.shape[0]
            t_h = t_face.shape[1]
            t_face = cv2.cvtColor(t_face, cv2.COLOR_BGR2GRAY)
            t_face = cv2.cvtColor(t_face, cv2.COLOR_GRAY2BGR)
            print("Original size")
            print(t_w)
            print(t_h)
            t_final = cv2.resize(t_face, (692, 1024))
            out_img = overlay_transparent(t_final, overlay_out, 0, 0)
            imgcopy = np.uint8(out_img)

            ran_papa = random.randint(1, 5)
            m_Text = ""
            if ran_papa == 1:
              m_Text = "爸爸我愛你！！！"
            elif ran_papa == 2:
              m_Text = "父親節快樂！！！"
            elif ran_papa == 3:
              m_Text = "爸爸節快樂！！！"
            elif ran_papa == 4:
              m_Text = "付清節快樂！！！"
            elif ran_papa == 5:
              m_Text = "爸爸永遠快樂！！！"

            ran_R = random.randint(0, 255)
            ran_G = random.randint(0, 255)
            ran_B = random.randint(0, 255)


            imgcopy = cv2ImgAddText(imgcopy, m_Text, 0, 715, (ran_R, ran_G, ran_B), 40)
            cv2.imwrite(dist_path, imgcopy)
        elif numFaces == 1:
            overlay_god_1 = cv2.imread("./tp_god_1.png", -1)
            overlay_god_2 = cv2.imread("./tp_god_2.png", -1)
            overlay_god_3 = cv2.imread("./tp_god_3.png", -1)
            overlay_god_4 = cv2.imread("./tp_god_4.png", -1)
            overlay_god_5 = cv2.imread("./tp_god_5.png", -1)
            overlay_out = overlay_god_1

            for face in faces:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                r = max(w, h)
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)
                
                cv2.rectangle(t_face, (nx, ny), (nx+nr,ny+nr), (0, 255, 0), 4, 4)
                t_cropface = t_face[ny:ny+nr, nx:nx+nr]
                shape = predictor(t_gray, face)
                shape = face_utils.shape_to_np(shape)

                for (s, t) in shape:
                   cv2.circle(t_face, (s, t), 1, (0, 0, 255), -1)

                incl = calculate_inclination(shape[17], shape[26])
                overlay_god_3 = rotate_bound(overlay_god_3, incl)
                (overlay_god_3, y_final) = adjust_sprite2head(overlay_god_3, 2*r, ny, True)
                t_cropface = draw_sprite(t_cropface,overlay_god_3, nx, y_final)

                t_resizeface = cv2.resize(t_cropface, (400, 400))

                blank_image = np.zeros((1024, 692, 3), np.uint8)
                blank_image[236:236 + t_resizeface.shape[0], 132:132 + t_resizeface.shape[1]] = t_resizeface
            out_img = overlay_transparent(blank_image, overlay_out, 0, 0)
            imgcopy = np.uint8(out_img)

            ran_god = random.randint(1, 5)
            m_Text = ""
            if ran_god == 1:
              m_Text = "恭喜發財！！！"
            elif ran_god == 2:
              m_Text = "紅包拿來！！！"
            elif ran_god == 3:
              m_Text = "財源滾滾來！！！"
            elif ran_god == 4:
              m_Text = "一路發發發！！！"
            elif ran_god == 5:
              m_Text = "好彩頭！！！"

            ran_R = random.randint(0, 255)
            ran_G = random.randint(0, 255)
            ran_B = random.randint(0, 255)

            imgcopy = cv2ImgAddText(imgcopy, m_Text, 190, 850, (ran_R, ran_G, ran_B), 55)
            cv2.imwrite(dist_path, imgcopy)
            print("Face 1 Scene2")
        elif numFaces >= 2:
            ran_number = random.randint(1, 2)
            #160 160 360 470
            if ran_number == 2:
                overlay_mama = cv2.imread("./tp_mama.png", -1)
                overlay_out = overlay_mama
                t_w = t_face.shape[0]
                t_h = t_face.shape[1]

                t_resizeface = cv2.resize(t_face, (360, 470))


                blank_image = np.zeros((1024, 692, 3), np.uint8)
                blank_image[160:160 + t_resizeface.shape[0], 160:160 + t_resizeface.shape[1]] = t_resizeface

                out_img = overlay_transparent(blank_image, overlay_out, 0, 0)
                cv2.imwrite(dist_path, out_img)

            else:
                overlay_comic = cv2.imread("./tp_comic.png", -1)
                overlay_out = overlay_comic
                t_w = t_face.shape[0]
                t_h = t_face.shape[1]
                t_face = cv2.cvtColor(t_face, cv2.COLOR_BGR2GRAY)
                t_face = cv2.cvtColor(t_face, cv2.COLOR_GRAY2BGR)

                t_final = cv2.resize(t_face, (692, 1024))
                out_img = overlay_transparent(t_final, overlay_out, 0, 0)
                imgcopy = np.uint8(out_img)

                ran_god = random.randint(1, 5)
                m_Text = ""
                if ran_god == 1:
                  m_Text = "漫畫測試1！！！"
                elif ran_god == 2:
                  m_Text = "漫畫測試2！！！"
                elif ran_god == 3:
                  m_Text = "漫畫測試3！！！"
                elif ran_god == 4:
                  m_Text = "漫畫測試4！！！"
                elif ran_god == 5:
                  m_Text = "漫畫測試5！！！"

                ran_R = random.randint(0, 255)
                ran_G = random.randint(0, 255)
                ran_B = random.randint(0, 255)

                imgcopy = cv2ImgAddText(imgcopy, m_Text, 130, 700, (ran_R, ran_G, ran_B), 55)
                cv2.imwrite(dist_path, imgcopy)


                # for face in faces: #if there are faces
            # *** Facial Landmarks detection
            # shape = predictor(gray, face)
            # shape = face_utils.shape_to_np(shape)

        # overlay_papa = cv2.imread("./tp_papa.png", -1)
        # overlay_mama = cv2.imread("./tp_mama.png", -1)
        # overlay_comic = cv2.imread("./tp_comic.png", -1)
        # overlay_god_1 = cv2.imread("./tp_god_1.png", -1)
        # overlay_god_2 = cv2.imread("./tp_god_2.png", -1)

        #t_final = cv2.resize(t_face, (692, 1024))
        #out_img = overlay_transparent(t_final, overlay_out, 0, 0)
        #cv2.imwrite(dist_path, out_img)

        try:
            client = ImgurClient(client_id, client_secret, access_token, refresh_token)
            config = {
                'album': album_id,
                'name': 'Catastrophe!',
                'title': 'Catastrophe!',
                'description': 'Cute kitten being cute on '
            }
            path = os.path.join('static', 'tmp', dist_name)
            output_image = client.upload_from_path(path, config=config, anon=False)
            os.remove(path)
            print(path)

            image_message = ImageSendMessage(
                original_content_url=output_image['link'],
                preview_image_url=output_image['link']
            )

            line_bot_api.reply_message(
                event.reply_token, image_message)

        except:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='上傳失敗'))
        return 0

    elif isinstance(event.message, VideoMessage):
        ext = 'mp4'
    elif isinstance(event.message, AudioMessage):
        ext = 'm4a'
    elif isinstance(event.message, TextMessage):
        ext = 'txt'
    else:
        return 0


if __name__ == '__main__':
    app.run()
