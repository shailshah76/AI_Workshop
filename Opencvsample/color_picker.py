import cv2

colors=[]
def on_mouse_click(event, x,y, flags, frame):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        colors.append(frame[y,x].tolist())
        
def get_color_range():
    cap=cv2.VideoCapture(0)
    while(len(colors)<10):
        ret_val,frame=cap.read()#to read the dynamic input from the camera
        if ret_val==True:
            hsv_img=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("HSV_COLOR_PICKER",hsv_img)
            cv2.setMouseCallback("HSV_COLOR_PICKER", on_mouse_click, hsv_img)
            #works without 0xFF n waittime is used for delay between two frames
            if cv2.waitKey(1) == 27:#when 27 is pressed loop ends
                break
    cv2.destroyAllWindows()
    cap.release()
    min_val=[]
    max_val=[]
    for i in range(0,3):
        min_val.append(min(x[i] for x in colors))
        max_val.append(max(x[i] for x in colors))
        
    return min_val,max_val

if __name__ == '__main__':
    min_val,max_val = get_color_range()
    
