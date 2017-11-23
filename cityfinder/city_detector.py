import cv2
import numpy as np
import PIL




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

    def find_pink_blob(self, path, out_path=None, imshow=False):
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if imshow:
            show_image('hsv', hsv)

        # mask pink:
        config = self.config['pink_blob']
        mask = np.logical_and(config['h'][0] < hsv[:, :, :1], hsv[:, :, :1] < config['h'][1])  # TODO add threshold by config['s']
        mask = 255 * mask.astype(np.uint8)
        # mask = np.repeat(mask, 3, axis=2)
        if imshow:
            show_image('pink', mask)

        # detect blobs in mask:
        blob_params = cv2.SimpleBlobDetector_Params()  # https://docs.opencv.org/trunk/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html
        # TODO adapt params, currently only finds small circular blobs: https://stackoverflow.com/questions/39083360/why-cant-i-do-blob-detection-on-this-binary-image
        detector = cv2.SimpleBlobDetector_create(blob_params)
        mask = cv2.bitwise_not(mask)
        blobs = detector.detect(mask)
        mask = cv2.bitwise_not(mask)

        print('detected %d blobs' % len(blobs))
        for blob in blobs:
            print(blob.pt, blob.size)

        im_with_keypoints = cv2.drawKeypoints(mask, blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if imshow:
            show_image('blobs', im_with_keypoints)

        if out_path is not None:
            cv2.imwrite(out_path, mask)

        return blobs

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
