from PIL import Image, ImageDraw, ImageFilter, ImageFont
from random import randint
from string import ascii_letters

width, height = (178, 218)
AMOUNT_OF_DATA = 202599
max_rectangle = 90
max_letter_length = 30


def random_string():
    res = ''
    for _ in range(randint(1, max_letter_length)):
        res += ascii_letters[randint(0, len(ascii_letters) - 1)]
    return res


def spoil(name, save_name):
    color = '#FFFFFF'
    changes = False
    im = Image.open(name)
    drawer = ImageDraw.Draw(im)
    blur = randint(0, 2)
    if randint(0, 1):
        blur = 0
    if blur != 0 and randint(0, 1) == 1:
        im = im.filter(ImageFilter.GaussianBlur(radius=blur))
        changes = True
    if randint(0, 1) == 1:
        first, second = randint(0, width), randint(0, height)
        drawer.ellipse((
            (first, second),
            (first + randint(1, max_rectangle), randint(1, max_rectangle))),
                        color)
        changes = True
    if randint(0, 1) == 1:
        first, second = randint(0, width), randint(0, height)
        drawer.rectangle((
            (first, second),
            (first + randint(1, max_rectangle), randint(1, max_rectangle))),
                        color)
        changes = True
    if True:  # randint(0, 2) == 1:
        font = ImageFont.truetype("arial.ttf", randint(20, 50))
        drawer.text((randint(1, width), randint(1, height)), random_string(), font=font)
        changes = True

    if not(changes):
        spoil(name, save_name)
    else:
        im.save(save_name)


def main(AMOUNT_OF_DATA):
    for i in range(1, AMOUNT_OF_DATA + 1):
        name = (6 - len(str(i))) * '0' + str(i) + '.jpg'
        spoil('datasets\\dataset\\' + name, 'datasets\\dataset_spoiled\\' + name)


if __name__ == '__main__':
    main(AMOUNT_OF_DATA)
