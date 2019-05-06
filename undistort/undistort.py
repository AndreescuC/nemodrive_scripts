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


def get_camera_matrix():
    return array([[1173.122620, 0.000000, 969.335924], [0.000000, 1179.612539, 549.524382], [0., 0., 1.]])


def get_distortion():
    return array([0.053314, -0.117603, -0.004064, -0.001819, 0.000000])


def get_distortion_2():
    return array([0.034238, -0.089798, 0.001695, -0.003938, 0.000000])


def get_distortion_3():
    return array([0.036627, -0.100169, -0.000988, -0.001462, 0.000000])


def main():
    img = cv2.imread("/home/andi/Sandbox/AIMAS/Scripts/landmark_visualizer/andia/rvec_logs/25_nov/session_1/1543145789.03_camera_3_off_180ms.jpg")


    h, w = img.shape[:2]
    mtx = get_camera_matrix()
    dist = get_distortion()

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('Images/calibresult.png', dst)

    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, None, (w, h), 5)
    # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('Images/calibresult2.png', dst)


if __name__ == '__main__':
    main()

