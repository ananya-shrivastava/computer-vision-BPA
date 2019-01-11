
# coding: utf-8

# In[1]:


import cv2
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if ret == True:

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',frame)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.imwrite('C:/Users/hi/AppData/Local/Programs/Python/Python36/models/object_detection/test_images/image90.jpg',frame)
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

