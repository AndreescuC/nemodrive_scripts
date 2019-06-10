def predict_projected_cv(self, img_row, img_col, segmentation_h, segmentation_w):
    img_col = 158
    img_row = 325
    u, v = self.convert_real_resolution(img_col, img_row, segmentation_h, segmentation_w)

    image_coordinates = np.array([u, v, 1]).reshape(3, 1)

    left_s_m = (self.ce_aux_inv_R @ self.ci_aux_inv_M) @ image_coordinates
    right_s_m = self.ce_aux_inv_R @ self.ce_translation

    s = (0 + right_s_m[1][0]) / left_s_m[1][0]
    world_coordinates = self.ce_aux_inv_R @ (s * self.ci_aux_inv_M @ image_coordinates - self.ce_translation)

    return world_coordinates[0], world_coordinates[2]


def predict_projected(self, img_row, img_col, img_height, img_width):
    # Predict real world y (depth)
    if img_row > img_height / 2:
        img_row = img_row - img_height / 2
    cam_projection_ratio = abs(2 * img_row / img_height)
    world_y = self.ce_translation['height'] / \
              tan(atan(cam_projection_ratio * tan(self.camera_vertical_fov / 2)) + self.ce_rotation['pitch'])

    # Predict real world x (side)
    if img_col > img_width / 2:
        img_col = img_col - img_width / 2
    cam_projection_ratio = abs(2 * img_col / img_width)
    world_x = world_y * (
            cam_projection_ratio * tan(self.camera_horizontal_fov / 2) * cos(self.ce_rotation['yaw']) +
            sin(self.ce_rotation['yaw'])
    )

    return world_y, world_x


def predict_projected_cv_2(self, img_row, img_col, segmentation_h, segmentation_w):

    image_coord = np.array([img_col, img_row, 1]).reshape(3, 1)

    R = np.array([
        [1, 0, 0],
        [0, 0, 1.6],
        [0, 1, 0]
    ])

    s = np.array([
        [640 / 1920, 0, 0],
        [0, 360 / 1080, 0],
        [0, 0, 1]
    ])
    M = np.linalg.inv(s @ self.ci_camera_matrix)

    wc = np.linalg.inv(R) @ M @ image_coord

    wc = wc / wc[2]
    return wc[0], wc[1]

# def prep(self):
#     imagePoints = [[271., 109.], [65., 208.], [334., 459.], [600., 225.]]
#     objectPoints = [0., 0., 0.], [-511., 2181., 0.], [-3574., 2354., 0.], [-3400., 0., 0.]
#
#     cameraMatrix = self.ci_camera_matrix
#     self.ci_aux_inv_M = np.linalg.inv(cameraMatrix)
#
#     _ret, rvec, tvec = cv2.solvePnP(np.float32(objectPoints), np.float32(imagePoints), np.float32(cameraMatrix), np.float32(self.ci_distortion))
#     self.ce_translation = np.array(tvec).reshape(3, 1)
#     rotationMatrix = cv2.Rodrigues(rvec)[0]
#     self.ce_aux_inv_R = np.linalg.inv(rotationMatrix)
#     u = 363
#     v = 222
#
#     uvPoint = [u, v, 1]
#
#
#     # my code
#     image_coordinates = np.array(uvPoint).reshape(3, 1)
#
#     left_s_m = self.ce_aux_inv_R @ self.ci_aux_inv_M @ image_coordinates
#     right_s_m = self.ce_aux_inv_R @ self.ce_translation
#
#     s = (285 + right_s_m[2][0]) / left_s_m[2][0]
#     world_coordinates = self.ce_aux_inv_R @ (s * self.ci_aux_inv_M @ image_coordinates - self.ce_translation)
#
#     return world_coordinates[0], world_coordinates[1]


def project_segmentation(self, segmentation):
    projection = deepcopy(self.frame)
    segmentation_h, segmentation_w, _ = segmentation.shape
    projection_h, projection_w = projection.shape

    for row, row_elements in enumerate(segmentation):
        for column, pixel in enumerate(row_elements):

            # not part of segmentation
            if pixel.tolist() == [0, 0, 0]:
                continue

            # x, y = self.predict_projected(row, column, segmentation_h, segmentation_w)
            x, y = self.predict_projected_cv_2(row, column, segmentation_h, segmentation_w)

            # grey segmentation
            projection[self.to_pixel_height(y[0], projection_h)][self.to_pixel_width(x[0], projection_w)] = 127

    draw_projection(projection)
    return projection


def blit_transform(surf, image, pos, origin_pos, real_shape, dimension, scale_factor, angle):
    w, h = image.get_size()

    pos = scale_position(pos, scale_factor, dimension, real_shape)
    origin_pos = scale_position(origin_pos, scale_factor, dimension, real_shape)

    scaled_origin_pos = (origin_pos[0] * scale_factor, origin_pos[0] * scale_factor)

    crop_origin = [scaled_origin_pos[i] - 400 if scaled_origin_pos[i] >= 400 else 0 for i in range(2)]
    max_crop = (min(800, w - scaled_origin_pos[0]), min(800, h - scaled_origin_pos[1]))
    cropped_image = image.subsurface(pygame.Rect(crop_origin, max_crop))

    # compute the axis aligned bounding box of the rotated image
    # box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    # box_rotate = [p.rotate(angle) for p in box]
    # min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    # max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # compute the translation of the pivot
    # pivot = pygame.math.Vector2(scaled_origin_pos[0], -scaled_origin_pos[1])
    # pivot_rotate = pivot.rotate(angle)
    # pivot_move = pivot_rotate - pivot



    # compute blit-ing point
    # width_from_center = dimension / 2 - pos[0]
    # height_from_center = dimension / 2 - pos[0]
    # origin = (0, 0)
    #     pos[0] - scaled_origin_pos[0] + min_box[0] - pivot_move[0] + width_from_center,
    #     pos[1] - scaled_origin_pos[1] - max_box[1] + pivot_move[1] + height_from_center
    # )

    # rotated_image = pygame.transform.rotate(cropped_image, 45)

    # surf.blit(rotated_image, origin)

    cropped_w, cropped_h = max_crop
    cropped_origin_pos = (cropped_w / 2, cropped_h / 2)

    # compute the axis aligned bounding box of the rotated image
    box = [pygame.math.Vector2(p) for p in [(0, 0), (cropped_w, 0), (cropped_w, -cropped_h), (0, -cropped_h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # compute the translation of the pivot
    pivot = pygame.math.Vector2(cropped_origin_pos[0], -cropped_origin_pos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # compute blit-ing point
    origin = (
        min_box[0] - pivot_move[0],
        -max_box[1] + pivot_move[1]
    )

    rotated_image = pygame.transform.rotate(cropped_image, angle)
    surf.blit(rotated_image, origin)
