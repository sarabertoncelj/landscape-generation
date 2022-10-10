from __future__ import division

import cv2
import scipy
import numpy as np
from imutils import face_utils
import socket
import tensorflow as tf
import dlib


def eye_aspect_ratio(eye):
    A = scipy.spatial.distance.euclidean(eye[1], eye[5])
    B = scipy.spatial.distance.euclidean(eye[2], eye[4])
    C = scipy.spatialdistance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def run() :
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5065

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    shape_x = 48
    shape_y = 48

    name_json = 'model.json'
    name_h5 = 'weights.h5'
    json_file = open(name_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(name_h5)
    print("Loaded model from disk")

    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor("face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    video_capture = cv2.VideoCapture(0)

    while True:
        # shrani frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

        for (i, rect) in enumerate(rects):

            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)

            #obrezemo obraz

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h,x:x+w]

            if (face.shape[0] != 0 and face.shape[1] != 0):
                face = scipy.ndimage.zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            else:
                face = scipy.ndimage.zoom(face, (shape_x, shape_y))

            face = face.astype(np.float32)

            #Scale
            try:
                face /= float(face.max())
                face = np.reshape(face.flatten(), (1, shape_x, shape_y, 1))

                # uporabimo model za napoved
                prediction = model.predict(face)
                prediction_result = np.argmax(prediction)

                cv2.rectangle(frame, (0, 0), (640, 40), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, 'Za izhod pritisni e', (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 10, 0), 2)

                cv2.putText(frame, "Obraz " + str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (240, 255, 240), 2)

                for (j, k) in shape:
                    cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)

                cv2.putText(frame, "----------------", (40, 100 + 180 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (240, 255, 240), 0)
                cv2.putText(frame, "Custva na obrazu st." + str(i + 1), (40, 120 + 180 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (240, 255, 240), 0)
                cv2.putText(frame, "Jeza : " + str(round(prediction[0][0], 3)), (40, 140 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 0)
                cv2.putText(frame, "Gnus : " + str(round(prediction[0][1], 3)), (40, 160 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 0)
                cv2.putText(frame, "Strah : " + str(round(prediction[0][2], 3)), (40, 180 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 1)
                cv2.putText(frame, "Sreca : " + str(round(prediction[0][3], 3)), (40, 200 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 1)
                cv2.putText(frame, "Zalost : " + str(round(prediction[0][4], 3)), (40, 220 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 1)
                cv2.putText(frame, "Presenecenje : " + str(round(prediction[0][5], 3)), (40, 240 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 1)
                cv2.putText(frame, "Neutralno : " + str(round(prediction[0][6], 3)), (40, 260 + 180 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 1)

                print("Custva na obrazu st." + str(i + 1))
                print("Jeza : " + str(round(prediction[0][0], 3)))
                print("Gnus : " + str(round(prediction[0][1], 3)))
                print("Strah : " + str(round(prediction[0][2], 3)))
                print("Sreca : " + str(round(prediction[0][3], 3)))
                print("Zalost : " + str(round(prediction[0][4], 3)))
                print("Presenecenje : " + str(round(prediction[0][5], 3)))
                print("Neutralno : " + str(round(prediction[0][6], 3)))

                if prediction_result == 0:
                    cv2.putText(frame, "Jeza", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)
                elif prediction_result == 1:
                    cv2.putText(frame, "Gnus", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)
                elif prediction_result == 2:
                    cv2.putText(frame, "Strah", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)
                elif prediction_result == 3:
                    cv2.putText(frame, "Sreca", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)
                elif prediction_result == 4:
                    cv2.putText(frame, "Zalost", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)
                elif prediction_result == 5:
                    cv2.putText(frame, "Presenecenje", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)
                else:
                    cv2.putText(frame, "Neutralno", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 0), 2)

                sock.sendto(("" + str(round(prediction[0][0], 3)) + " " + str(round(prediction[0][1], 3)) + " " + str(
                    round(prediction[0][2], 3)) + " " + str(round(prediction[0][3], 3)) + " " + str(
                    round(prediction[0][4], 3)) + " " + str(round(prediction[0][5], 3)) + " " + str(
                    round(prediction[0][6], 3))).encode(), (UDP_IP, UDP_PORT))

            except:
                print("exc")

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (240, 255, 240), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (240, 255, 240), 1)
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [noseHull], -1, (240, 255, 240), 1)
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (240, 255, 240), 1)
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [jawHull], -1, (240, 255, 240), 1)
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(frame, [ebrHull], -1, (240, 255, 240), 1)
            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(frame, [eblHull], -1, (240, 255, 240), 1)
            #
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            video_capture.release()
            cv2.destroyAllWindows()
            break



def main():
    run()

if __name__ == "__main__":

    main()
