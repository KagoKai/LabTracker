import cv2

def draw_overlays(cv2_im, objs, labels, arr_dur, arr_track_data):
    height, width, channels = cv2_im.shape
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    global tolerance
    
    #draw black rectangle on top
    cv2_im = cv2.rectangle(cv2_im, (0,0), (width, 24), (0,0,0), -1)
    
     
    #write processing durations
    cam=round(arr_dur[0]*1000,0)
    inference=round(arr_dur[1]*1000,0)
    other=round(arr_dur[2]*1000,0)
    text_dur = 'Camera: {}ms   Inference: {}ms   other: {}ms'.format(cam,inference,other)
    cv2_im = cv2.putText(cv2_im, text_dur, (int(width/4)-30, 16),font, 0.4, (255, 255, 255), 1)
    
    #write FPS 
    total_duration=cam+inference+other
    fps=round(1000/total_duration,1)
    text1 = 'FPS: {}'.format(fps)
    cv2_im = cv2.putText(cv2_im, text1, (10, 20),font, 0.7, (150, 150, 255), 2)
   
    
    #draw black rectangle at bottom
    cv2_im = cv2.rectangle(cv2_im, (0,height-24), (width, height), (0,0,0), -1)
    
    #write deviations and tolerance
    str_tol='Tol : {}'.format(tolerance)
    cv2_im = cv2.putText(cv2_im, str_tol, (10, height-8),font, 0.55, (150, 150, 255), 2)
   
   
    x_dev=arr_track_data[2]
    str_x='X: {}'.format(x_dev)
    if(abs(x_dev)<tolerance):
        color_x=(0,255,0)
    else:
        color_x=(0,0,255)
    cv2_im = cv2.putText(cv2_im, str_x, (110, height-8),font, 0.55, color_x, 2)
    
    y_dev=arr_track_data[3]
    str_y='Y: {}'.format(y_dev)
    if(abs(y_dev)<tolerance):
        color_y=(0,255,0)
    else:
        color_y=(0,0,255)
    cv2_im = cv2.putText(cv2_im, str_y, (220, height-8),font, 0.55, color_y, 2)
    
    
    #write direction, speed, tracking status
    cmd=arr_track_data[4]
    cv2_im = cv2.putText(cv2_im, str(cmd), (int(width/2) + 10, height-8),font, 0.68, (0, 255, 255), 2)
    
    delay1=arr_track_data[5]
    str_sp='Speed: {}%'.format(round(delay1/(0.1)*100,1))
    cv2_im = cv2.putText(cv2_im, str_sp, (int(width/2) + 185, height-8),font, 0.55, (150, 150, 255), 2)
    
    if(cmd==0):
        str1="No object"
    elif(cmd=='Stop'):
        str1='Acquired'
    else:
        str1='Tracking'
    cv2_im = cv2.putText(cv2_im, str1, (width-140, 18),font, 0.7, (0, 255, 255), 2)
    
    #draw center cross lines
    cv2_im = cv2.rectangle(cv2_im, (0,int(height/2)-1), (width, int(height/2)+1), (255,0,0), -1)
    cv2_im = cv2.rectangle(cv2_im, (int(width/2)-1,0), (int(width/2)+1,height), (255,0,0), -1)
    
    #draw the center red dot on the object
    cv2_im = cv2.circle(cv2_im, (int(arr_track_data[0]*width),int(arr_track_data[1]*height)), 7, (0,0,255), -1)

    #draw the tolerance box
    cv2_im = cv2.rectangle(cv2_im, (int(width/2-tolerance*width),int(height/2-tolerance*height)), (int(width/2+tolerance*width),int(height/2+tolerance*height)), (0,255,0), 2)
    
    #draw bounding boxes
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        
        box_color, text_color, thickness=(0,150,255), (0,255,0),2
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, thickness)
        

        #text3 = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        #cv2_im = cv2.putText(cv2_im, text3, (x0, y1-5),font, 0.5, text_color, thickness)
        
    return cv2_im