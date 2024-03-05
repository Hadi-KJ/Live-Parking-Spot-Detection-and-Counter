import cv2
from util import get_parking_spots_bboxes, empty_or_not
import numpy as np 
import matplotlib.pyplot as plt

# just a slight approach to get the difference of two images
def calc_diff(im1, im2):
  return np.abs(np.mean(im1) - np.mean(im2))

mask = "./mask_1920_1080.png"
video_path = "./samples/parking_1920_1080_loop.mp4"

mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

# making a graph of the spots using mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
#getting the bounding box of spots by using the elements of each component
spots = get_parking_spots_bboxes(connected_components)
#print(spots[0])

spots_status = [None for j in spots]
diffs = [None for j in spots]
# to get the difference with current frame so that we avoid classifying every frame
previous_frame = None

ret = True
frame_num = 0
step = 90

while ret:
  ret, frame = cap.read()

  ############ to check the difference with previous frame ############
  if frame_num % step ==0 and previous_frame is not None:
    for spot_index, spot in enumerate(spots):
      x1, y1, w, h = spot
      spot_crop = frame[y1:y1 + h, x1:x1+w, :]
      
      diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1+w, :])

    #print([diffs[j] for j in np.argsort(diffs)][::-1])
    #print(len(diffs)) #396
    #plt.figure()
    #plt.hist([diffs[j]/np.amax(diffs) for j in np.argsort(diffs)][::-1])
    #if frame_num == 300:
    #  plt.show()
    #based on the histograms we realize that only a few high values have changes in every check
    # which means there are a few changes in spots

  ############ check the spots every 30 frames ############
  if frame_num % step ==0:
    if previous_frame is None:
      Max_diffs_index = range(len(spots))
    else:
      Max_diffs_index = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs) > 0.4]
    
    for spot_index in Max_diffs_index:
      x1, y1, w, h = spots[spot_index]

      spot_crop = frame[y1:y1 + h, x1:x1+w, :]
      spots_status[spot_index] = empty_or_not(spot_crop)
  
  if frame_num % step ==0 :
    previous_frame = frame.copy()

  ############ To draw the reactangle in video ############
  for spot_index, spot in enumerate(spots):
    spot_status = spots_status[spot_index]
    x1, y1, w, h = spots[spot_index]
    if spot_status:
      frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 255, 0), 2)
    else:
      frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0, 0, 255), 2)

  cv2.rectangle(frame, (80,20), (550, 80), (0,0,0), -1)
  cv2.putText(frame, 'Available spots: {}/{}'.format(str(sum(spots_status)), str(len(spots_status))), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

  cv2.imshow("frame", frame)
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break

  if frame_num==30:
    cv2.imwrite("example.jpg", frame)
    print("Image saved successfully.")
    
  frame_num +=1

cap.release()
cv2.destroyAllWindows()