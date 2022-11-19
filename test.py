from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename,askdirectory
# import the necessary packages
from imutils import perspective
import numpy as np
import cv2

import matplotlib.pyplot as plt


colors = {'red': (0, 0, 255), 'white': (255, 255, 255),'blue':(212,120,0)}


# rename persian digits to english digits
def rename_persian_digits_to_english_digits(text):
    persian_digits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
    english_digits = [str(i) for i in range(10)]
    for i in range(10):
        text = text.replace(persian_digits[i], english_digits[i])
    return text

# rename and save file
def rename_and_save_file(file_path, new_file_path):
    file_path.rename(new_file_path)
    return new_file_path
# make directory
def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

# matlotlib show image
def show_image(image):
    plt.imshow(image)
    plt.show()

def get_file_path():
    Tk().withdraw()
    file_path = askopenfilename()
    return Path(file_path)


# list file in directory
def list_files_in_dir(suffix = ''):
    Tk().withdraw()
    file_path = askdirectory()
    return list(Path(file_path).glob("*{}".format(suffix)))


def load_image(file_path):
    return cv2.imread(str(file_path))

# Check labels count
def check_labes_count(label,n = 4):
    if len(label) == n:
        return True
    return False

def draw_circle(event, x, y, flags, param):
    """Mouse callback function"""

    global circles
    
    if event == cv2.EVENT_LBUTTONDBLCLK and not check_labes_count(label):
        circles.append((x, y))
        nLabel = len(circles)
        label.append(nLabel)
        print("Add Label {}, (x={} , y={})".format(nLabel,x,y))
        
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # Delete last label
        nLabel = len(circles)
        if nLabel > 0:
            rX,rY = circles[-1]
            print("Remove Label {}, (x={} , y={})".format(nLabel,rX,rY))
        try:
            circles.pop()
            label.pop()
        except (IndexError):
            print("No label for remove!")

# Crop image
def crop_image(image, points):
    # extract the ordered coordinates, then apply the perspective
    warped = perspective.four_point_transform(image, np.array(points))
    return warped


# Structure to hold the created circles & labels:
circles = []
label = []
# Label number font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1

# # rename_persian_digits_to_english_digits
# for file in list_files_in_dir():
#     print(file)
#     new_file_path = file.parent / rename_persian_digits_to_english_digits(file.name)
#     rename_and_save_file(file, new_file_path)


fileId = 0
for file in list_files_in_dir(".jpg"):
    if fileId == 0:
        make_dir(Path(file.parent / 'croped'))
    # rename file
    new_file_path = file.parent / rename_persian_digits_to_english_digits(file.name)
    file = rename_and_save_file(file, new_file_path)
    
    fileId += 1
    notecard = load_image(file)
    croped = []
    id = 1
    show = True
    # Initialize the list of reference points and boolean indicating
    # Create a named window
    cv2.namedWindow('Mini map', cv2.WINDOW_NORMAL)

    # Set the mouse callback function
    cv2.setMouseCallback('Mini map', draw_circle)    
    while show:
        # in this case, image is temp image
        image = notecard.copy()
        
        # We draw now only the current circles:
        for index,pos in enumerate(circles):
            x,y=pos
            cv2.circle(image, pos, 10, colors['red'], -1)
            cv2.putText(image, str(label[index]), (x-5,y+5), font, font_scale, colors['white'], thickness)
        # Show image 'Mini map':
        cv2.imshow('Mini map', image)

        k = cv2.waitKey(200) & 0xFF
        
        if k == ord('p') and check_labes_count(label):
            print(label)
            print(circles)
            croped = crop_image(notecard ,circles)
            show_image(cv2.cvtColor(croped, cv2.COLOR_RGB2BGR))
            
        if k == ord('r'):
            circles = []
            label = []
            croped = []
            print("All Labels removed!")
            
        if k == ord('s') and len(croped):
            uniq = str(fileId) + "_" + str(id) + str(file.suffix)
            id += 1
            # Save croped image
            cv2.imwrite(str(file.parent / "croped" / uniq), croped)
            print("Save Image cropped!")
            circles = []
            label = []
            croped = []
        
        if k == ord('w') and check_labes_count(label):
            croped = crop_image(notecard ,circles)
            # show_image(cv2.cvtColor(croped, cv2.COLOR_RGB2BGR))
            uniq = str(fileId) + "_" + str(id) + str(file.suffix)
            id += 1
            # Save croped image
            cv2.imwrite(str(file.parent / "croped" / uniq), croped)
            print("Save Image cropped!")
            circles = []
            label = []
            croped = []
                        
        if k == ord('n'):
            circles = []
            label = []
            croped = []
            print("New Image!")
            # Destroy windows
            cv2.destroyAllWindows()
            show = False
        
        if k == ord('l'):
            print("Exit!")
            # Destroy windows
            cv2.destroyAllWindows()
            exit()