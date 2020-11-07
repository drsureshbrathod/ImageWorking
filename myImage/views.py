from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from .models import Hotel
from PIL import Image
import cv2
import os
import  numpy as np
from PIL import Image, ImageEnhance, ImageFilter
# Create your views here.
def hotel_image_view(request):
    if request.method == 'POST':
        form = HotelForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = HotelForm()
    return render(request, 'index.html', {'form': form})


def success(request):
    return HttpResponse('successfully uploaded')

def display_transpose_hotel_images(request):
    if request.method == 'GET':
        # getting all the objects of hotel.
        trans = Transpose.objects.all()
        return render(request,'display.html',{'hotel_images': trans})

def display_transpose_backface_images(request):
    if request.method == 'GET':
        # getting all the objects of hotel.
        trans = BackRemove.objects.all()
        return render(request,'display.html',{'hotel_images': trans})


def display_hotel_images(request):
    if request.method == 'GET':
        # getting all the objects of hotel.
        Hotels = Hotel.objects.all()
        return render(request,'display.html',{'hotel_images': Hotels})

def tryuse(request):
        if request.method == 'GET':
            img = cv2.imread(r"C:\Users\suresh\PycharmProjects\ImageWorking\media\images\original.jpg")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            img_contours = sorted(img_contours, key=cv2.contourArea)
            print("ok")
            for i in img_contours:
                if cv2.contourArea(i) > 100:
                    break
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [i], -1, 255, -1)
            new_img = cv2.bitwise_and(img, img, mask=mask)
            file_name = "back" + '.png'
            cv2.imwrite('C:\\Users\\suresh\\PycharmProjects\\ImageWorking\\media\images\\back.png', new_img)
            t = BackRemove()
            t.name = 'back'
            t.hotel_Main_Img = 'C:\\Users\suresh\\PycharmProjects\\ImageWorking\media\\images\\back.png'
            t.save()
            return display_transpose_backface_images(request)

        return  success(request)

def newTryImage(request):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    image_bgr = cv2.imread(r"C:\Users\suresh\PycharmProjects\ImageWorking\media\images\6a346c4c0d96554347f843a3515a1386.jpg")
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Rectange values: start x, start y, width, height
    rectangle = (0, 56, 256, 150)
    # Create initial mask
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run grabCut
    cv2.grabCut(image_rgb,  # Our image
                mask,  # The Mask
                rectangle,  # Our rectangle
                bgdModel,  # Temporary array for background
                fgdModel,  # Temporary array for background
                5,  # Number of iterations
                cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply image with new mask to subtract background
    image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
    cv2.imwrite("C:\\Users\suresh\\PycharmProjects\\ImageWorking\media\\images\\" + "nn.jpg", image_rgb_nobg)
    return  success(request)
    #plt.imshow(image_rgb_nobg), plt.axis("off")
    #plt.show()
def removeBack(path,threshold):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    mask = cv2.drawContours(threshed, cnt, 0, (0, 255, 0), 0)
    masked_data = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(cnt)
    dst = masked_data[y: y + h, x: x + w]

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(dst)
    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)
    return  dst

def newtry(path):
        # Load the Image
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)

    for i in img_contours:
        if cv2.contourArea(i) > 100:
                break
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [i], -1, 255, -1)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    return  new_img


def backRemoval(request):
    path = r"C:\Users\suresh\PycharmProjects\ImageWorking\media\images\6a346c4c0d96554347f843a3515a1386.jpg"
    #image= removeBack(path,259.)
    image= newtry(path)
    cv2.imwrite("C:\\Users\suresh\\PycharmProjects\\ImageWorking\media\\images\\"+"bb.jpg", image)
   # t = BackRemove()
 #   t.name = 'back'
   # t.hotel_Main_Img = 'C:\\Users\suresh\\PycharmProjects\\ImageWorking\media\\images\\bb.jpg'
   # t.save()
   # printos.path.join(fruit_class_path, image_name)
    #im = Image.open(r"C:\Users\suresh\PycharmProjects\ImageWorking\media\images\original.jpg")  # input image
    #im = im.filter(ImageFilter.MedianFilter())
    #enhancer = ImageEnhance.Contrast(im)
    #im = enhancer.enhance(2)
    #im = im.convert('1')
    #im.save('image_clear.jpg')  # ouput image
    return success(request)
def dotransperarnt(request):
    if request.method == 'GET':
        img = Image.open(r"C:\Users\suresh\PycharmProjects\ImageWorking\media\images\original.jpg")
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                if item[0] > 150:
                    newData.append((0, 0, 0, 255))
                else:
                    newData.append(item)
                    print(item)
        img.putdata(newData)
        img.save('C:\\Users\suresh\\PycharmProjects\\ImageWorking\media\\images\\Trans.png', "PNG")
        t= Transpose()
        t.name='trans'
        t.hotel_Main_Img='C:\\Users\suresh\\PycharmProjects\\ImageWorking\media\\images\\Trans.png'
        t.save()
        return display_transpose_hotel_images(request)
        #return render(request, 'display.html', {'hotel_images': Hotel})
