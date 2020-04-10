import numpy as np

# import our captioner
from captioner import *

def main():
    """Get output from captioner and assign to color values"""
    
    #get generated colors
    #color_input = caption_img()
    color_input = ['FFFFFF', '014367', '843875', '123904', '234765', '005533']

    # there are 16 colors to assign plus foreground, background, and cursor.
    cols = [''] * 19
    lines = [''] * 19

    # figure out how many colors there are
    # assign the colors
    # the first two are always white, black
    cols[0] = 'FFFFFF'
    cols[1] = '000000'
    count = 2
    for col in color_input:
        #assign the rest of the colors
        cols[count] = col
        count += 1

    count = 0
    for col in cols:
        print("line:", count, "color:", col)
        
        if count == 0:
            lines[0] =  "*.foreground:" + "\t#"+cols[0]

        elif count == 1:
            lines[1] =  "*.background:" + "\t#"+cols[1]

        elif count == 2:
            lines[2] =  "*.cursorColor:" + "\t#"+cols[2]

        lines[count] = "*.color"+str(count)+":" + "\t#"+col
        count += 1

    with open('Xresources', 'w') as f:
        f.writelines("%s\n" % l for l in lines)

if __name__ == "__main__":
    main()
