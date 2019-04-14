import sys

import matplotlib.pyplot as plt
from numpy import zeros, uint8, sqrt, arccos, pi as PI, cos, deg2rad
from skimage import io, measure, draw

IMAGE_SUFFIX = '.png'
POLYGON_TOLERANCE = 7
STRAIGHT_ANGLE = 90.0
FULL_ANGLE = 360.0
HALF_ANGLE = 180.0
ZERO_ANGLE = 0.0
STRAIGHT_ANGLE_TOLERANCE = 5.0
INNER = 'INNER'
OUTER = 'OUTER'
MAX_ANGLE_DIFFERENCES = 20.0
MAX_SECTION_RATIO_DIFFERENCES = 0.2
SECTION_WEIGHT = 0.5
ANGLE_WEIGHT = 0.5


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


def isInnerAngle(pointLeft, pointRight, image):
    return INNER \
        if image[int((pointLeft[0] + pointRight[0]) / 2), int((pointLeft[1] + pointRight[1]) / 2)] > 0 \
        else OUTER


def countAngles(points, image):
    return [[list(points[i]), [countAngle(points[i - 1], points[i], points[i + 1] if i < len(points) - 1 else points[0])
        , isInnerAngle(points[i - 1], points[i + 1] if i < len(points) - 1 else points[0], image)]]
            # [[posX, posY],angle]
            for i in range(0, len(points))]


def isAngleStraight(angle, addition):
    return abs(angle - STRAIGHT_ANGLE) < STRAIGHT_ANGLE_TOLERANCE + addition


def canBeBase(angle1, angle2, addition):
    return (angle1[1] == INNER and angle2[1] == INNER) and (
            isAngleStraight(angle1[0], addition) and isAngleStraight(angle2[0], addition))


def chooseBestBase(potentialBases):
    if len(potentialBases) == 1:
        return potentialBases[0]
    error = [((pBase[0][1][0] - STRAIGHT_ANGLE) ** 2 + (pBase[1][1][0] - STRAIGHT_ANGLE) ** 2) / 2 for pBase in
             potentialBases]
    return [potentialBases[i] for i, e in enumerate(error) if e == min(error)][0]


def turnAngleVector(angles, basePosition):
    if basePosition[0] == 0 and basePosition[1] == 1:
        return angles
    start = basePosition[0] if basePosition[1] - basePosition[0] == 1 else basePosition[1]
    return [angles[(i + start) % len(angles)] for i in range(len(angles))]


def prepareVector(points):
    potentialBase = []
    addition = 0
    while len(potentialBase) == 0:
        potentialBase = [[points[i], points[i + 1]] for i in range(-1, len(points) - 1) if
                         canBeBase(points[i][1], points[i + 1][1], addition)]
        addition += 1
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


def countMinimalSectionRatio(section1, section2):
    return section1 / section2 if section1 < section2 else section2 / section1


def prepareImagesDataVector(images):
    # return [[i, countDistances(prepareVector(countAngles(findPolygon(image), image)))] for i, image in enumerate(images)]
    imagesData = []
    for i, image in enumerate(images):
        polygon = findPolygon(image)
        withAngles = countAngles(polygon, image)
        print("Angles = ", withAngles)
        startBasePoints = prepareVector(withAngles)
        print("Start base angles = ", startBasePoints)
        withDistances = countDistances(startBasePoints)
        print("With distances = ", withDistances)
        imagesData.append([i, withDistances])
        # item = [imageIndex, [ [ [pointX, pointY], [angle, INNER/OUTER angle], [distanceLeft, distanceRight] ]... ]
    print(imagesData)
    return imagesData


# -(1/MAX
def compareAngle(angleReference, anglePoint):
    difference = MAX_ANGLE_DIFFERENCES
    if angleReference[1] == INNER and anglePoint[1] == INNER:
        difference = abs(HALF_ANGLE - (angleReference[0] + anglePoint[0]))
    elif (angleReference[1] == INNER and anglePoint[1] == OUTER) \
            or (angleReference[1] == OUTER and anglePoint[1] == INNER):
        difference = abs(ZERO_ANGLE - (angleReference[0] - anglePoint[0]))
    elif angleReference[1] == OUTER and anglePoint[1] == OUTER:
        difference = abs(FULL_ANGLE - (angleReference[0] + anglePoint[0]))
    return max(0.0, -(1.0 / MAX_ANGLE_DIFFERENCES) * difference + 1.0)


# -(1/MAX_SECTION_RATIO_DIFFERENCES) * x + 1 or 0, where x is difference between ratios
def compareSection(sectionReference, sectionPoint):
    referenceRatio = countMinimalSectionRatio(sectionReference[0], sectionReference[1])
    pointRatio = countMinimalSectionRatio(sectionPoint[0], sectionPoint[1])
    return max(0.0, -(1.0 / MAX_SECTION_RATIO_DIFFERENCES) * (abs(referenceRatio - pointRatio)) + 1.0)


# [ [pointX, pointY], [angle, INNER/OUTER angle], [distanceLeft, distanceRight] ]
def compare2Points(referencePoint, point):
    sectionRatio = compareSection(referencePoint[2], point[2])
    angleRatio = compareAngle(referencePoint[1], point[1])
    return SECTION_WEIGHT * sectionRatio + ANGLE_WEIGHT * angleRatio


def join2Points(point1, point2):  # [[joined angle, joined angle/2], section]
    angle1 = point1[1][0] if point1[1][0] == INNER else FULL_ANGLE - point1[1][0]
    angle2 = point2[1][0] if point2[1][0] == INNER else FULL_ANGLE - point2[1][0]
    avgAngle = (angle1 + angle2) / 2
    halfAvgAngle = avgAngle / 2
    section = sqrt(point1[2][1] ** 2 + point2[2][0] ** 2 + 2 * point1[2][1] * point2[2][0] * cos(deg2rad(angle1)))
    return [[avgAngle, halfAvgAngle], section]


def preparePairPoints(points):  # list of [[joined angle, joined angle/2], section]
    return [join2Points(points[i], points[i + 1]) for i in range(len(points) - 1)]


def countSimilarity(reference, imageData):
    similarityLeft = 0
    similarityRight = 0
    if len(reference) == len(imageData):
        for i in range(0, len(reference)):
            similarityLeft += compare2Points(reference[i], imageData[i])
            similarityRight += compare2Points(reference[i], imageData[len(imageData) - 1 - i])
    else:
        bigger = reference if len(reference) > len(imageData) else imageData
        smaller = reference if len(reference) < len(imageData) else imageData
        joinPoints = preparePairPoints(bigger)
        # todo
        print(len(reference), len(imageData), len(joinPoints), joinPoints)
    return max(similarityLeft, similarityRight)


def createSimilarities(imagesData):
    similarities = [(reference[0], [(j, countSimilarity(reference[1][2:], imageData[1][2:]))  # cutting off base points
                                    for j, imageData in enumerate(imagesData) if i != j])
                    for i, reference in enumerate(imagesData)]
    print(similarities)
    return similarities


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    images = readImages(pathToImages, imageNumber)
    imagesData = prepareImagesDataVector(images)
    similarities = createSimilarities(imagesData)


if __name__ == '__main__':
    main()
