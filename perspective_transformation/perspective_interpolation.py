from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import yaml
import numpy as np
import random


TRAIN_TEST_SPLIT_FACTOR = 0.8
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
SHUFFLE_AUTOMATIC = True


def split_data_set(c1_points, c2_points):
    assert TRAIN_TEST_SPLIT_FACTOR < 1

    limit = int(TRAIN_TEST_SPLIT_FACTOR * c1_points.shape[0])

    c1_points = np.array_split(c1_points, [limit])
    c2_points = np.array_split(c2_points, [limit])

    return c1_points[0], c1_points[1], c2_points[0], c2_points[1]


def load_config():
    with open("config/selection_points.yaml", 'r') as stream:
    # with open("config/manual_points.yaml", 'r') as stream:

        c1_points = c2_points = []
        img1 = img2 = ''

        try:
            conf = yaml.load(stream)
            matching_mode = conf['matching']

            if matching_mode in ['manual', 'combined', 'learn_automatic_test_manual']:
                c1_points = conf['c1']['points']
                c2_points = conf['c2']['points']

            if matching_mode in ['automatic', 'combined', 'learn_automatic_test_manual']:
                img1 = conf['c1']['image']
                img2 = conf['c2']['image']

        except yaml.YAMLError as exc:
            print(exc)
            return {}

    return {
        'matching_mode': matching_mode,
        'image1': img1,
        'image2': img2,
        'points1': c1_points,
        'points2': c2_points
    }


def evaluate(estimated_points, real_points, labels):
    errors = []
    for estimated, real in zip(estimated_points, real_points.values):
        error_Y = abs(estimated[0] - real[0])
        error_X = abs(estimated[1] - real[1])

        print(
            "Errors [%f  %f] ( %s )" %
            (
                error_Y,
                error_X,
                labels[(real[0], real[1])] if (real[0], real[1]) in labels else "AUTOMATICALLY GENERATED POINT"
            )
        )
        errors.append((error_Y, error_X))

    return errors


def get_matches_automatically(img_path1, img_path2):

    if not img_path1 or not img_path2:
        return np.array([]), np.array([])

    im1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    if not SHUFFLE_AUTOMATIC:
        matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("images/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2


def shuffle(points1, points2):
    pairs = [(p1, p2) for p1, p2 in zip(points1.tolist(), points2.tolist())]
    random.shuffle(pairs)

    return \
        np.array([pair[0] for pair in pairs], dtype='float32'),\
        np.array([pair[1] for pair in pairs], dtype='float32')


def main():
    cfg = load_config()

    c1_labels = {(point_y, point_x): description for point_y, point_x, description in cfg['points1']}
    c2_labels = {(point_y, point_x): description for point_y, point_x, description in cfg['points2']}

    c1_points = np.array([[point_y, point_x] for point_y, point_x, description in cfg['points1']], dtype='float32')
    c2_points = np.array([[point_y, point_x] for point_y, point_x, description in cfg['points2']], dtype='float32')
    detected_points_1, detected_points_2 = get_matches_automatically(cfg['image1'], cfg['image2'])

    bulk_c1_points = np.array(c1_points.tolist() + detected_points_1.tolist(), dtype='float32')
    bulk_c2_points = np.array(c2_points.tolist() + detected_points_2.tolist(), dtype='float32')
    shuffled_c1_points, shuffled_c2_points = shuffle(bulk_c1_points, bulk_c2_points)

    df_x = pd.DataFrame(shuffled_c1_points, columns=['Y', 'X'])
    df_y = pd.DataFrame(shuffled_c2_points, columns=['Y', 'X'])

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

    l_reg = linear_model.LinearRegression()
    l_reg.fit(x_train, y_train)
    c2_estimated = l_reg.predict(x_test)

    estimation_errors = evaluate(c2_estimated, y_test, c2_labels)
    print("\n")
    print("Average error: (%f, %f)" % (
        sum([e[0] for e in estimation_errors]) / len(y_test),
        sum([e[1] for e in estimation_errors]) / len(y_test)
    ))


class InterpolationModel:

    def __init__(self, full_train=False):
        self.full_train = full_train
        self.cfg = load_config()
        self.model = None
        self.trained = False
        self.trained_on = None

    def get_name(self):
        return "InterpolationModel_%s" % self.model.__class__.__name__

    def train(self):
        c1_labels = {(point_y, point_x): description for point_y, point_x, description in self.cfg['points1']}
        c2_labels = {(point_y, point_x): description for point_y, point_x, description in self.cfg['points2']}

        c1_points = np.array([[point_y, point_x] for point_y, point_x, description in self.cfg['points1']], dtype='float32')
        c2_points = np.array([[point_y, point_x] for point_y, point_x, description in self.cfg['points2']], dtype='float32')
        detected_points_1, detected_points_2 = get_matches_automatically(self.cfg['image1'], self.cfg['image2'])

        bulk_c1_points = np.array(c1_points.tolist() + detected_points_1.tolist(), dtype='float32')
        bulk_c2_points = np.array(c2_points.tolist() + detected_points_2.tolist(), dtype='float32')
        shuffled_c1_points, shuffled_c2_points = shuffle(bulk_c1_points, bulk_c2_points)

        self.trained_on = shuffled_c1_points
        self.actual_points = shuffled_c2_points

        df_x = pd.DataFrame(shuffled_c1_points, columns=['Y', 'X'])
        df_y = pd.DataFrame(shuffled_c2_points, columns=['Y', 'X'])

        if self.full_train:
            x_train = df_x
            y_train = df_y
        else:
            x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

        self.model = linear_model.LinearRegression()
        self.model.fit(x_train, y_train)
        self.trained = True

        if not self.full_train:
            c2_estimated = self.model.predict(x_test)
            estimation_errors = evaluate(c2_estimated, y_test, c2_labels)
            print("\n")
            print("Average error: (%f, %f)" % (
                sum([e[0] for e in estimation_errors]) / len(y_test),
                sum([e[1] for e in estimation_errors]) / len(y_test)
            ))

    def predict_points(self, points):
        assert self.trained
        to_be_predicted = pd.DataFrame(np.array(points), columns=['Y', 'X'])
        return self.model.predict(to_be_predicted)


if __name__ == '__main__':
    main()
