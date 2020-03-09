from sklearn.cluster import KMeans
from collections import Counter
import cv2
import sys

def read_image(filepath):
    image = cv2.imread(filepath)
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

def save_colors(colors, filename):
    f = open(filename, 'w')

    with open(filename, 'w') as f:
        for item in colors:
            f.write("%s\n" % item)

if __name__ == "__main__":
    colors = get_color(read_image(sys.argv[1]), 8)
    print_colors(colors)
    parts = sys.argv[1].split("/")
    fname = parts[len(parts)-1].split(".")
    save_colors(colors, "img/training/palettes/"+fname[0]+".txt")
