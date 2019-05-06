import cv2
import yaml
from perspective_interpolation import InterpolationModel
from naive_model import IdentityModel, NaiveTranslationModel
from transform_perspective import HomographyModel


def load_config():
    with open("config/points_translation_config.yaml", 'r') as stream:

        try:
            conf = yaml.load(stream)
            color = conf['annotation_color']
            color_tolerance = conf['annotation_color_tolerance']
            annotated_image = conf['annotated']
            to_be_annotated_image = conf['to_be_annotated']
            result_image = conf['result']

        except yaml.YAMLError as exc:
            print(exc)
            return {}

    return {
        'annotation_color': color,
        'color_tolerance': color_tolerance,
        'annotated': annotated_image,
        'to_be_annotated': to_be_annotated_image,
        'result': result_image
    }


def is_annotation_color(color1, color2, color_tolerance):
    assert len(color1) == len(color2)
    color1_rgb_format = color1.tolist()
    color1_rgb_format.reverse()
    return sum([abs(component1-component2) for component1, component2 in zip(color1_rgb_format, color2)]) <= color_tolerance


def mark_as_actual_road(current_color):
    return [255, current_color[1], current_color[2]]


def mark_as_road(current_color):
    return [current_color[0], 255, current_color[2]]


def translate_annotation(annotated, to_be_annotated, annotation_color, color_tolerance):
    assert annotated.shape == to_be_annotated.shape

    model = InterpolationModel(full_train=True)
    # model = NaiveTranslationModel(delta_x=15)
    # model = IdentityModel()
    # model = HomographyModel(full_train=True)
    model.train()
    #model.warp_perspective(annotated)

    height, width, _ = annotated.shape
    to_be_translated = []
    for y in range(height):
        for x in range(width):
            if not is_annotation_color(annotated[y][x], list(annotation_color.values()), color_tolerance):
                continue
            to_be_translated.append([x, y])

    outside = []
    translated_points = model.predict_points(to_be_translated)
    for point in translated_points.tolist():
        x = int(point[0])
        y = int(point[1])

        if y >= height or x >= width or y < 0 or x < 0:
            outside.append((y, x))
            continue

        to_be_annotated[y][x] = mark_as_road(to_be_annotated[y][x])

    return to_be_annotated, model.get_name()


def main():
    cfg = load_config()

    annotated = cv2.imread(cfg['annotated'], cv2.IMREAD_COLOR)
    to_be_annotated = cv2.imread(cfg['to_be_annotated'], cv2.IMREAD_COLOR)

    to_be_annotated, name = translate_annotation(annotated, to_be_annotated, cfg['annotation_color'], cfg['color_tolerance'])

    cv2.imwrite("%s%s.png" % (cfg['result'], name), to_be_annotated)


if __name__ == '__main__':
    main()
