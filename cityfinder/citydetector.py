import cv2
import numpy as np
import PIL




class CityDetector:
    def __init__(self, config=None, debug_path=None):
        self.config = config or self.default_config()
        self.debug_path = debug_path

    def find_circle_city(self, path, out_path=None, imshow=False):
        img = cv2.imread(path)
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

    def find_pink_blob(self, path, out_path=None, imshow=False):
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if imshow:
            show_image('hsv', hsv)

        config = self.config['pink_blob']
        mask = np.logical_and(config['h'][0] < hsv[:, :, :1], hsv[:, :, :1] < config['h'][1])  # TODO add threshold by config['s']
        # mask = np.repeat(mask, 3, axis=2)

        if imshow:
            show_image('pink', 255 * mask.astype(np.uint8))
        if out_path is not None:
            cv2.imwrite(out_path, hsv)

        return None

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
            }
        }
        return config


def show_image(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
