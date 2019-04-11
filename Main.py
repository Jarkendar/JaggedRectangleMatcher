import sys
from skimage import io, measure, draw
import matplotlib.pyplot as plt
from numpy import zeros, shape, uint8

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


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    images = readImages(pathToImages, imageNumber)

    for i in images:
        polygon = findPolygon(i)
        drawPolygon(polygon, shape(i))


if __name__ == '__main__':
    main()
