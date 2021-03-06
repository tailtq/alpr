## Network

**Note:** This kind of network doesn't follow YOLO, SSD, or RCNN detection layers for multiple objects detection. So the output is just one object. Therefore, it's hard to add new labels to teach network ideas about car's windows, wheels...


## Loss function

Utility: For measuring the difference between warped ground-truth and prediction. 

**3 small losses:**
- Object probability loss **(log - cross entropy loss)**
- Non-object probability loss **(log - cross entropy loss)**
- Bounding box loss (warped state) **(L1 loss)**

**Steps:**
- Calculate the Object and Non-object probability losses
- Having the transformation values, calculate the warped version of canonical square by transforming it in each pixel
- Calculate the bounding box loss (pixel level)

According to the paper, the author said: "first part of the loss
function considers the error between a warped version of the canonical square
and the normalized annotated points of the LP" (Sergio, 2018, p. 8)

