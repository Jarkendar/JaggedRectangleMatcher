import sys
from itertools import combinations

import matplotlib.pyplot as plt
from numpy import zeros, uint8, sqrt, arccos, pi as PI
from skimage import io, measure, draw

#CONSTANT
IMAGE_SUFFIX = '.png'
STRAIGHT_ANGLE = 90.0
FULL_ANGLE = 360.0
HALF_ANGLE = 180.0
ZERO_ANGLE = 0.0
INNER = 'INNER'
OUTER = 'OUTER'

#PARAMS
POLYGON_TOLERANCE = 9  # more = faster
STRAIGHT_ANGLE_TOLERANCE = 5.0  # no influence
MAX_ANGLE_DIFFERENCES = 14.0  # best 14 in range 10-20
MAX_SECTION_RATIO_DIFFERENCES = 0.1  # best 0.1 in range 0.05-0.2
SECTION_WEIGHT = 1.0
ANGLE_WEIGHT = 2.125  # best in range 0.25-4.0

#VARIABLES DEPENDENT FROM PARAMS
INVERSE_MAX_ANGLE_DIFFERENCES = 1.0 / MAX_ANGLE_DIFFERENCES
INVERSE_MAX_SECTION_RATIO_DIFFERENCES = 1.0 / MAX_SECTION_RATIO_DIFFERENCES


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


def isInnerAngle(pointLeft, pointCenter, pointRight, image):
    betweenX = (pointLeft[0] + pointRight[0]) / 2
    betweenY = (pointLeft[1] + pointRight[1]) / 2
    return INNER \
        if image[int((pointCenter[0] + betweenX) / 2), int((pointCenter[1] + betweenY) / 2)] > 0 \
        else OUTER


def countAngles(points, image):
    return [[list(points[i]), [countAngle(points[i - 1], points[i], points[i + 1] if i < len(points) - 1 else points[0])
        , isInnerAngle(points[i - 1], points[i], points[i + 1] if i < len(points) - 1 else points[0], image)]]
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
    bestPotentialBase = chooseBestBase(potentialBase)
    angleBasePosition = [i for i in range(len(points)) if points[i] in bestPotentialBase]
    turnedVector = turnAngleVector(points, angleBasePosition)
    return turnedVector


def calcDist(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def countDistances(points):
    return [[points[i][0], points[i][1],
             [calcDist(points[i][0], points[i - 1][0]),
              calcDist(points[i][0], points[(i + 1) % len(points)][0]),
              countMinimalSectionRatio(calcDist(points[i][0], points[i - 1][0]),
              calcDist(points[i][0], points[(i + 1) % len(points)][0]))]
             ] for i in range(len(points))]


def countMinimalSectionRatio(section1, section2):
    return section1 / section2 if section1 < section2 else section2 / section1


def prepareImagesDataVector(images):
    imagesData = []
    for i, image in enumerate(images):
        polygon = findPolygon(image)
        withAngles = countAngles(polygon, image)
        startBasePoints = prepareVector(withAngles)
        withDistances = countDistances(startBasePoints)
        imagesData.append([i, withDistances])
        # item -> [imageIndex, [ [ [pointX, pointY], [angle, INNER/OUTER angle], [distanceLeft, distanceRight] ]... ]
    return imagesData


# -(1/MAX_ANGLE_DIFFERENCES) * x + 1 or 0, where x is difference between angles
def compareAngle(angleReference, anglePoint):
    if angleReference[1] == INNER and anglePoint[1] == INNER:
        difference = abs(HALF_ANGLE - (angleReference[0] + anglePoint[0]))
    elif angleReference[1] == OUTER and anglePoint[1] == OUTER:
        difference = abs(FULL_ANGLE - (angleReference[0] + anglePoint[0]))
    else:
        difference = abs(ZERO_ANGLE - (angleReference[0] - anglePoint[0]))
    return 0.0 if difference > MAX_ANGLE_DIFFERENCES else -INVERSE_MAX_ANGLE_DIFFERENCES * difference + 1.0


# -(1/MAX_SECTION_RATIO_DIFFERENCES) * x + 1 or 0, where x is difference between ratios
def compareSection(sectionReference, sectionPoint):
    difference = abs(sectionReference[2] - sectionPoint[2])
    return 0.0 if difference > MAX_SECTION_RATIO_DIFFERENCES else -INVERSE_MAX_SECTION_RATIO_DIFFERENCES * difference + 1.0


# [ [pointX, pointY], [angle, INNER/OUTER angle], [distanceLeft, distanceRight] ]
def compare2Points(referencePoint, point):
    return SECTION_WEIGHT * compareSection(referencePoint[2], point[2]) \
           + ANGLE_WEIGHT * compareAngle(referencePoint[1], point[1])


def join2Points(point1Left, point1, point2, point2Right):  # [[joined angle, INNER/OUTER], [sectionLeft, sectionRight]]
    avgPoint = [(point1[0][0] + point2[0][0]) / 2.0, (point1[0][1] + point2[0][1]) / 2.0]
    angle = countAngle(point1Left[0], avgPoint, point2Right[0])
    angleLocality = INNER
    if point1[1][1] == point2[1][1]:
        angleLocality = point1[1][1]
    elif point1[1][0] > point2[1][0]:
        angleLocality = point2[1][1]
    elif point1[1][0] <= point2[1][0]:
        angleLocality = point1[1][1]

    sectionLeft = calcDist(point1Left[0], avgPoint)
    sectionRight = calcDist(avgPoint, point2Right[0])
    return [avgPoint, [angle, angleLocality],
            [sectionLeft, sectionRight, countMinimalSectionRatio(sectionLeft, sectionRight)]]


def preparePairPoints(points):  # list of [[joined angle, joined angle/2], section]
    return [join2Points(points[i - 1], points[i], points[(i + 1) % len(points)], points[(i + 2) % len(points)])
            for i in range(len(points) - 1)]


def buildSmallerSizePointList(combination, bigger, joinPair):
    copy = bigger.copy()
    for i in combination:
        del copy[i]
        del copy[i]
        copy.insert(i, joinPair[i])
    return copy


def countAvgSimilarity(smaller, bigger, joinPair):
    sumSimilarityLeft = 0.0
    sumSimilarityRight = 0.0
    i = 0
    for i, combination in enumerate(combinations(range(len(smaller)), len(bigger) - len(smaller))):
        biggerSmaller = buildSmallerSizePointList(combination, bigger, joinPair)
        for j in range(0, len(smaller)):
            sumSimilarityLeft += compare2Points(smaller[j], biggerSmaller[j])
            sumSimilarityRight += compare2Points(smaller[j], biggerSmaller[- 1 - j])
    divider = i if i > 0.0 else 1.0
    return sumSimilarityLeft / divider, sumSimilarityRight / divider


def countSimilarity(reference, imageData):
    similarityLeft = 0
    similarityRight = 0
    if len(reference) == len(imageData):
        for i in range(len(reference)):
            similarityLeft += compare2Points(reference[i], imageData[i])
            similarityRight += compare2Points(reference[i], imageData[- 1 - i])
    else:
        bigger = reference if len(reference) > len(imageData) else imageData
        smaller = reference if len(reference) < len(imageData) else imageData
        joinPoints = preparePairPoints(bigger)
        similarityLeft, similarityRight = countAvgSimilarity(smaller, bigger, joinPoints)
    return max(similarityLeft, similarityRight)


def createSimilarities(imagesData):
    similarities = [(reference[0], [(j, countSimilarity(reference[1][2:], imageData[1][2:]))  # cutting off base points
                                    for j, imageData in enumerate(imagesData) if i != j])
                    for i, reference in enumerate(imagesData)]
    return similarities


def createRanking(similarities):
    ranking = []
    for objectSimilarity in similarities:
        objectSimilarity[1].sort(key=lambda tup: tup[1], reverse=True)
        ranking.append([objectSimilarity[0], [t[0] for t in objectSimilarity[1]]])
    ranking.sort(key=lambda t: t[0])
    return ranking


def printRanking(ranking):
    print('\n'.join([' '.join([str(i) for i in item[1]]) for item in ranking]))


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    images = readImages(pathToImages, imageNumber)
    imagesData = prepareImagesDataVector(images)
    similarities = createSimilarities(imagesData)
    ranking = createRanking(similarities)
    printRanking(ranking)


if __name__ == '__main__':
    main()
