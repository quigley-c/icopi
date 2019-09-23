from sklearn.cluster import KMeans
import cv2
from collections import Counter

def read_image():
    #TODO: Change to input path
    image = cv2.imread('../img/sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
#end

def get_color(image, nocolors):
    image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = KMeans(n_clusters = nocolors)
    labels = clf.fit_predict(image)

    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return rgb_colors
#end

def print_colors(colors):
    print("colors in order of percent are: \n")
    for color in colors:
        print(color, "\n")
#end

print_colors(get_color(read_image(), 8))


