import sys

import matplotlib.pyplot as plt
from numpy import zeros, uint8, sqrt, arccos, pi as PI
from skimage import io, measure, draw

IMAGE_SUFFIX = '.png'
POLYGON_TOLERANCE = 7


def readImages(path, numberOfImages):
    return [io.imread(path + '/' + str(i) + IMAGE_SUFFIX) for i in range(numberOfImages)]


def showImage(image):
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.show()


def drawPolygon(polygon, shape):
    rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1])
    img = zeros(shape, dtype=uint8)
    img[rr, cc] = 1
    showImage(img)


def findPolygon(image):
    bestContour = None
    for contour in measure.find_contours(image, 0):
        if bestContour is None or len(bestContour) < len(contour):
            bestContour = contour
    return measure.approximate_polygon(bestContour, POLYGON_TOLERANCE)


def countAngle(pLeft, pCenter, pRight):
    a = (pCenter[0] - pLeft[0]) ** 2 + (pCenter[1] - pLeft[1]) ** 2
    b = (pCenter[0] - pRight[0]) ** 2 + (pCenter[1] - pRight[1]) ** 2
    c = (pRight[0] - pLeft[0]) ** 2 + (pRight[1] - pLeft[1]) ** 2
    return arccos((a + b - c) / sqrt(4 * a * b)) * 180 / PI


def countAngles(points):
    return [countAngle(points[i - 1], points[i], points[i + 1] if i < len(points) - 1 else points[1])
            for i in range(1, len(points))]


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    images = readImages(pathToImages, imageNumber)

    for i in images:
        polygon = findPolygon(i)
        angles = countAngles(polygon)


if __name__ == '__main__':
    main()
