"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from multiprocessing import Process, Queue

import numpy as np
from datetime import datetime

import cv2

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer


def tmp_log(array,now):
    TIME_INFO_LEN = 5
    array=np.reshape(array,(3,))
    log_array = np.empty((1,array.shape[0]+TIME_INFO_LEN))
    log_array[0,:array.shape[0]]=array
    log_array[0,array.shape[0]:]=\
            np.array([now.day,now.hour,now.minute,now.second,now.microsecond])
    return log_array

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()
CNN_INPUT_SIZE = 128

class HeadposeClass:
    def __init__(self,sample_frame,filename):
        self.log_filename = filename

        self.mark_detector = MarkDetector()

        self.img_queue = Queue()
        self.box_queue = Queue()

        self.img_queue.put(sample_frame)
        self.box_process = Process(target=self.get_face, args=())

        self.box_process.start()

        height, width = sample_frame.shape[:2]
        self.pose_estimator = PoseEstimator(img_size=(height, width))

        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

    def get_face(self):
        """Get face from image queue. This function is used for multiprocessing"""
        while True:
            image = self.img_queue.get()
            box = self.mark_detector.extract_cnn_facebox(image)
            self.box_queue.put(box)

    def get_euler(self,pose):
        _pose = np.reshape(pose, (-1, 3))
        rvec, tvec = _pose
        dst, jcobi = cv2.Rodrigues(rvec)
        dst2 = np.empty((3,4))
        dst2[:3,:3]=dst
        angles = cv2.decomposeProjectionMatrix(dst2)
        euler = angles[-1]

        #        yaw = euler[1]
        #        pitch = euler[0]
        #        roll = euler[2]

        return euler

    def send_pose(self,frame):
        data=self.ret_pose(frame)
        print("send",data)
        euler=self.get_euler(data)

        now_time = datetime.now()
        save_data = tmp_log(euler,now_time)
        np.savetxt(self.log_filename,save_data,fmt="%f",delimiter=",")

    def ret_pose(self,frame):
        self.img_queue.put(frame)
        facebox = self.box_queue.get()

        if facebox is not None:
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = self.mark_detector.detect_marks(face_img)

            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            pose = self.pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            stabile_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
            stabile_pose = np.reshape(stabile_pose, (-1, 3))
            return stabile_pose
        else:
            return np.zeros((2,3))

def get_headpose(filename):

    # Specify the camera which you want to use. The default argument is '0'
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_capture.set(cv2.CAP_PROP_FPS, 15)

    ret = None

    ret, frame = video_capture.read()
    headpose = HeadposeClass(frame,filename)

    while(True):
        ret, frame = video_capture.read()
        #cv2.imwrite(file, frame)
        if frame is not None:
            headpose.send_pose(frame)
            #data=headpose.ret_pose(frame)
            #print(data)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Image could not be captured")

if __name__ == '__main__':
    filename = "test_headpose_log.csv"
    get_headpose(filename)
