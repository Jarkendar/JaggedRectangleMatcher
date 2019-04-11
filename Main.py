import sys

import matplotlib.pyplot as plt
from numpy import zeros, uint8, sqrt, arccos, pi as PI
from skimage import io, measure, draw

IMAGE_SUFFIX = '.png'
POLYGON_TOLERANCE = 7
STRAIGHT_ANGLE = 90.0
STRAIGHT_ANGLE_TOLERANCE = 5.0


class Point:
    def __init__(self, point):
        self.coordinates = point
        self.leftNeighbor = None
        self.rightNeighbot = None
        self.angle = None
        self.leftDistance = 0.0
        self.rightDistance = 0.0


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


def isAngleStraight(angle):
    return abs(angle - STRAIGHT_ANGLE) < STRAIGHT_ANGLE_TOLERANCE


def canBeBase(angle1, angle2):
    return isAngleStraight(angle1) and isAngleStraight(angle2)


def chooseBestBase(potentialBases):
    if len(potentialBases) == 1:
        return potentialBases[0]
    error = [((pBase[0] - STRAIGHT_ANGLE) ** 2 + (pBase[1] - STRAIGHT_ANGLE) ** 2) / 2 for pBase in potentialBases]
    return [potentialBases[i] for i, e in enumerate(error) if e == min(error)][0]


def turnAngleVector(angles, basePosition):
    if basePosition[0] == 0 and basePosition[1] == 1:
        return angles
    start = basePosition[0] if basePosition[1] - basePosition[0] == 1 else basePosition[1]
    return [angles[(i + start) % len(angles)] for i in range(len(angles))]


def prepareAngleVector(angles):
    potentialBase = [[angles[i], angles[i + 1]] for i in range(-1, len(angles) - 1) if
                     canBeBase(angles[i], angles[i + 1])]
    print("Potential Bases = ", potentialBase)
    bestPotentialBase = chooseBestBase(potentialBase)
    print("Best potential base = ", bestPotentialBase)
    angleBasePosition = [i for i in range(len(angles)) if angles[i] in bestPotentialBase]
    print("Base position = ", angleBasePosition)
    turnedVector = turnAngleVector(angles, angleBasePosition)
    print("Turned vector = ", turnedVector)
    return turnedVector


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    images = readImages(pathToImages, imageNumber)

    for i in images:
        polygon = findPolygon(i)
        angles = countAngles(polygon)
        print("Angles = ", angles)
        startBaseAngles = prepareAngleVector(angles)
        print("Start base angles = ", startBaseAngles)
        print()


if __name__ == '__main__':
    main()
