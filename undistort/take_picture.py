import cv2
from numpy import array, matrix, ndarray


def get_camera_matrix_1():
    return array([[1122.873891, 0.000000, 954.962391], [0.000000, 1123.453009, 561.582887], [0.000000, 0.000000, 1.000000]])


def get_camera_matrix_2():
    return array([[1127.783600, 0.000000, 944.349473], [0.000000, 1128.385489, 581.754914], [0.000000, 0.000000, 1.000000]])


def get_camera_matrix_3():
    return array([[505.045106, 0.000000, 308.875448], [0.000000, 505.162560, 246.552722], [0.000000, 0.000000, 1.000000]])


def get_distortion_1():
    return array([0.049052, -0.114197, -0.000437, 0.002550, 0.000000])


def get_distortion_2():
    return array([0.034238, -0.089798, 0.001695, -0.003938, 0.000000])


def get_distortion_3():
    return array([0.036627, -0.100169, -0.000988, -0.001462, 0.000000])


def main():
    v = cv2.VideoCapture(1)
    for i in range(5):
        ret, frame = v.read()
        v.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        v.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

    while True:
        ret, frame = v.read()

        cv2.imshow("test", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

    cv2.imwrite("Images/test.jpg", frame)


if __name__ == '__main__':
    main()

