import os

import cv2
import numpy as np
import PIL



city_indices = {
    '0': 'tlv',
    '1': 'haifa',
    '2': 'netania',
    '3': 'gaza',
    '4': 'nazareth',
    '5': 'j-m',
    '6': 'aman',
    '7': 'kiryat gat',
    '8': 'tiberias',
    '9': 'kiryat shmona',
    '10': 'nablus',
}


class CityDetector:
    def __init__(self, config=None, debug_path=None):
        self.config = config or self.default_config()
        self.debug_path = debug_path

    def find_circle_city(self, path, out_path=None, imshow=False):
        img = cv2.imread(path, 0)
        # img = cv2.medianBlur(img, 5)

        config = self.config['hough_circle']
        circles = cv2.HoughCircles(img, **config)
        if circles is None or circles[0] is None or len(circles[0]) == 0:
            return []

        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # draw the center of the circle

        if imshow:
            show_image('detected circles',cimg)

        if out_path is not None:
            cv2.imwrite(out_path, cimg)
        return circles[0]

    def find_pink_blob(self, path, out_folder=None, imshow=False):
        img = cv2.imread(path)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, '1_rgb.png'), img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        os.makedirs(out_folder, exist_ok=True)
        if imshow:
            show_image('hsv', hsv)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, '2_hsv.png'), hsv)

        # mask pink:
        config = self.config['pink_blob']
        mask = np.logical_and(config['h'][0] < hsv[:, :, :1], hsv[:, :, :1] < config['h'][1])  # TODO add threshold by config['s']
        mask = 255 * mask.astype(np.uint8)
        # mask = np.repeat(mask, 3, axis=2)
        if imshow:
            show_image('pink', mask)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, '3_mask.png'), mask)

        # morphology:
        mask = cv2.medianBlur(mask, config['median_size'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((config['opening_radius'], config['opening_radius']), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((config['closing_radius'], config['closing_radius']), np.uint8))
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, '4_mask_closed.png'), mask)

        # detect blobs in mask:
        blob_params = cv2.SimpleBlobDetector_Params()  # https://docs.opencv.org/trunk/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html
        blob_params.filterByCircularity = False
        blob_params.filterByConvexity = False
        blob_params.filterByInertia = False
        blob_params.minDistBetweenBlobs = 100  # todo config
        blob_params.minArea = 200  # todo config
        blob_params.maxArea = 1e10
        # TODO adapt params, currently only finds small circular blobs: https://stackoverflow.com/questions/39083360/why-cant-i-do-blob-detection-on-this-binary-image
        detector = cv2.SimpleBlobDetector_create(blob_params)
        mask = cv2.bitwise_not(mask)
        blobs = detector.detect(mask)
        mask = cv2.bitwise_not(mask)

        print('detected %d blobs' % len(blobs))
        # for blob in blobs:
        #     print(blob.pt, blob.size)

        mask_with_blobs = cv2.drawKeypoints(mask, blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if imshow:
            show_image('blobs', mask_with_blobs)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, '5_blobs.png'), mask_with_blobs)

        # choose largest blobs:
        blobs = sorted(blobs, key=lambda k: k.size, reverse=True)
        # largest_blobs = blobs[:config['num_blobs']]
        largest_blobs = self.choose_largest_fartest_blobs(blobs)

        print('chose %d largest blobs' % len(largest_blobs))
        mask_with_largest_blobs = cv2.drawKeypoints(mask, largest_blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        w, h = mask.shape[:2]
        print('w,h:', w, h)
        largest_blobs = {city_indices[str(idx)]: (blob.pt[0], blob.pt[1]) for idx, blob in enumerate(largest_blobs)}
        for idx, blob in largest_blobs.items():
            print(str(idx))
            print(blob)
            x, y = blob
            cv2.putText(mask_with_largest_blobs, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0),2,cv2.LINE_AA)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, '6_largest_blobs.png'), mask_with_largest_blobs)

        largest_blobs = {name: (correct_x(x, w), h - y) for name, (x, y) in largest_blobs.items()}
        return largest_blobs

    def choose_largest_fartest_blobs(self, blobs):
        def are_blobs_close(blob1, blob2):
            dist = np.linalg.norm(np.asarray(blob1.pt) - np.asarray(blob2.pt))
            print('dist:', dist)
            return dist < self.config['pink_blob']['eliminate_closest_pix']

        blobs = sorted(blobs, key=lambda k: k.size, reverse=True)
        best = []
        while len(blobs) > 0:
            print('ITERATION')
            best_candidate = blobs[0]
            best.append(best_candidate)
            blobs = [b for b in blobs if not are_blobs_close(b, best_candidate)]
        return best


    @classmethod
    def default_config(cls):
        config = {
            'hough_circle': {
                'method': cv2.HOUGH_GRADIENT,
                'dp': 1,
                'minDist': 20,
                'param1': 50,
                'param2': 30,
                'minRadius': 0,
                'maxRadius': 20,
            },
            'pink_blob': {
                'h': [150, 180],
                's': [50, 60],
                'median_size': 5,
                'opening_radius': 3,
                'closing_radius': 21,
                'eliminate_closest_pix': 500,
                'num_blobs': 20,
            }
        }
        return config


def show_image(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def correct_x(x, width):
    factor = 0.66
    return width/2 + factor * (x - width/2)