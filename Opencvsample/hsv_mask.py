import cv2
import numpy as np

import color_picker as cp

def create_hsv_mask(min_val,max_val):
    cap=cv2.VideoCapture(0)
    while (1):
        ret_val, frame = cap.read()
        if ret_val == True:
            hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            lower_value=np.array(min_val)
            higher_value=np.array(max_val)
            mask=cv2.inRange(frame, lower_value, higher_value)
            and_img=cv2.bitwise_and(frame, frame, mask=mask)
            canny_img = cv2.Canny(frame,100,300)
            cv2.imshow("Canny_Image", canny_img)
            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()
    cap.realese()
if __name__ == '__main__':
    min_val,max_val=cp.get_color_range()
    create_hsv_mask(min_val, max_val)

