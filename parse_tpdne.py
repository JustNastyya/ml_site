import requests
from PIL import Image
from PIL import ImageChops
import os


def check_images():
    i = 792
    image_prev = Image.open('datasets\\dataset_unreal\\00792.png')
    for name in os.listdir('datasets\\dataset_unreal')[793:]:
        image_two = Image.open('datasets\\dataset_unreal\\' + name)
        if ImageChops.difference(image_prev, image_two).getbbox():
            image_two.save('datasets\\dataset_unreal\\' + (5 - len(str(i // 2))) * '0' + str(i) + '.png')
            i += 1
        image_prev = image_two
        if i % 50 == 0:
            print(i)
    print(i)




'''
image_one = Image.open(path_one)
image_two = Image.open(path_two)

diff = ImageChops.difference(image_one, image_two)

if diff.getbbox():
    print("images are different")
else:
    print("images are the same")

'''
def save_im(save_or_not, i):
    response = requests.get("https://thispersondoesnotexist.com/image")

    if save_or_not:
        name = (5 - len(str(i // 2))) * '0' + str(i) + '.png'
        file = open("datasets\\dataset_unreal\\" + name, "wb")
        file.write(response.content)
        file.close()


def main():
    save_or_not = 0
    for i in range(792, 1209):
        try:
            response = requests.get("https://thispersondoesnotexist.com/image")
            if save_or_not:
                name = (5 - len(str(i // 2))) * '0' + str(i) + '.png'
                file = open("datasets\\dataset_unreal\\" + name, "wb")
                file.write(response.content)
                file.close()
        except Exception as e:
            print('exception occured', e)
        save_or_not += 1
        save_or_not = save_or_not % 3
        if i % 100 == 0:
            print(i)


main()
print('main started')
check_images()