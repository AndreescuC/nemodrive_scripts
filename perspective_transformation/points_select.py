import numpy as np
import cv2
import yaml
from ruamel import yaml as ruamel_yaml
from ruamel.yaml import YAML


points = {}


def get_camera_params():
    K = np.array([
        [1173.122620, 0.000000, 969.335924],
        [0.000000, 1179.612539, 549.524382],
        [0.000000, 0.000000, 1.000000]
    ])
    S = np.array([
        [360. / 1080., 0., 0.],
        [0., 640. / 1920., 0.],
        [0., 0., 1.]
    ])
    K = np.matmul(S, K)

    return K, S


def load_config():
    with open("config/points_select_config.yaml", 'r') as stream:

        try:
            conf = ruamel_yaml.load(stream)
            images_no = conf['images']
            path = conf['path']
            log_file = conf['log_file']

            images = conf['images_info']
            assert len(images) == images_no

            global points
            points = {image_info['tag']: [] for image_info in images}
            files = {image_info['tag']: image_info['file'] for image_info in images}

        except ruamel_yaml.YAMLError as exc:
            print(exc)
            return {}

    return {
        'path': path,
        'log_path': log_file,
        'files': files
    }


def log_selected_points(log_file, paths):
    camera_index = 1
    entries = {'matching': "manual"}
    for tag, current_points in points.items():
        entries['c%d' % camera_index] = {
            'wat_image': '#%s' % tag,
            'image': paths[tag],
            'points': [[point[0], point[1], '#%d' % index] for index, point in enumerate(current_points)]
        }
        camera_index += 1

    with open(log_file, 'w+') as outfile:
        yaml.dump(entries, outfile)


def add_points(event, x, y, flags, params):
    global points

    modified_tag = params[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        points[modified_tag].append((x, y))
        for tag, current_points in points.items():
            print("%s points: " % tag, current_points)
        print("-------------------------------------")


def show_points(points, frame, window_name):
    if points:
        cv2.putText(frame, str(len(points)), points[len(points) - 1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(window_name, frame)
    cv2.imshow(window_name, frame)


def main():
    global points

    cfg = load_config()
    K, S = get_camera_params()

    paths = {tag: cfg['path'] + file_name for tag, file_name in cfg['files'].items()}

    images = [(tag, cv2.imread(cfg['path'] + file_name, cv2.IMREAD_COLOR)) for tag, file_name in cfg['files'].items()]
    images = [(image_info[0], cv2.undistort(image_info[1], K, 0)) for image_info in images]

    K[0, 2] += images[0][1].shape[1]

    while True:
        for tag, image in images:
            cv2.imshow(tag, image)
            cv2.namedWindow(tag)
            cv2.setMouseCallback(tag, add_points, param=[tag])
            show_points(points[tag], image, tag)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:                     # Escape exits the tool
            exit()
        elif k == 32:                   # Space
            log_selected_points(cfg['log_path'], paths)
            break


if __name__ == "__main__":
    main()
