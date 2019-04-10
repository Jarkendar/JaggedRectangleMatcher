import sys
from skimage import io

IMAGE_SUFFIX = '.png'


def readImages(path, numberOfImages):
    return [io.imread(path + '/' + str(i) + IMAGE_SUFFIX) for i in range(numberOfImages)]


def main():
    pathToImages = sys.argv[1]
    imageNumber = int(sys.argv[2])
    readImages(pathToImages, imageNumber)


if __name__ == '__main__':
    main()
