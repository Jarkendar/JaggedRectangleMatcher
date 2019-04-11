import sys

import matplotlib.pyplot as plt
from numpy import zeros, uint8, sqrt, arccos, pi as PI
from skimage import io, measure, draw

IMAGE_SUFFIX = '.png'
POLYGON_TOLERANCE = 7
STRAIGHT_ANGLE = 90.0
STRAIGHT_ANGLE_TOLERANCE = 5.0


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
    return measure.approximate_polygon(bestContour, POLYGON_TOLERANCE)[:-1]  # without last point (same as first)


def countAngle(pLeft, pCenter, pRight):
    a = (pCenter[0] - pLeft[0]) ** 2 + (pCenter[1] - pLeft[1]) ** 2
    b = (pCenter[0] - pRight[0]) ** 2 + (pCenter[1] - pRight[1]) ** 2
    c = (pRight[0] - pLeft[0]) ** 2 + (pRight[1] - pLeft[1]) ** 2
    return arccos((a + b - c) / sqrt(4 * a * b)) * 180 / PI


def countAngles(points):
    return [[list(points[i]), countAngle(points[i - 1], points[i], points[i + 1] if i < len(points) - 1 else points[0])]
            # [[posX, posY],angle]
            for i in range(0, len(points))]


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


def prepareVector(points):
    potentialBase = [[points[i], points[i + 1]] for i in range(-1, len(points) - 1) if
                     canBeBase(points[i][1], points[i + 1][1])]
    print("Potential Bases = ", potentialBase)
    bestPotentialBase = chooseBestBase(potentialBase)
    print("Best potential base = ", bestPotentialBase)
    angleBasePosition = [i for i in range(len(points)) if points[i] in bestPotentialBase]
    print("Base position = ", angleBasePosition)
    turnedVector = turnAngleVector(points, angleBasePosition)
    print("Turned vector = ", turnedVector)
    return turnedVector


def calcDist(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def countDistances(points):
    return [[points[i][0], points[i][1],
             [calcDist(points[i][0], points[i - 1][0]), calcDist(points[i][0], points[(i + 1) % len(points)][0])]]
            for i in range(len(points))]


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    images = readImages(pathToImages, imageNumber)
    imagesData = []

    for i, image in enumerate(images):
        polygon = findPolygon(image)
        withAngles = countAngles(polygon)
        print("Angles = ", withAngles)
        startBasePoints = prepareVector(withAngles)
        print("Start base angles = ", startBasePoints)
        withDistances = countDistances(startBasePoints)
        print("With distances = ", withDistances)
        imagesData.append([i, withDistances])
        # item = [imageIndex, [ [ [pointX, pointY], angle, [distanceLeft, distanceRight] ]... ]
    print(imagesData)


if __name__ == '__main__':
    main()
