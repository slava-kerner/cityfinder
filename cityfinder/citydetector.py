import cv2
import numpy as np
import PIL


class CityDetector:
    def __init__(self, config=None, debug_path=None):
        self.config = config or self.default_config()
        self.debug_path = debug_path

    def find_circle_city(self, path):
        img = cv2.imread(path, 0)
        img = cv2.medianBlur(img, 5)

        config = self.config['hough_circle']
        circles = cv2.HoughCircles(img, **config)
        print('Detected %d circles' % len(circles))
        # circles = cv2.HoughCircles(img, config['method'], config['dp'], config['mid_dist'],..)

        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow('detected circles',cimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
                'maxRadius': 0,
            },
        }
        return config
