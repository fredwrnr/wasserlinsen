from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO('runs/segment/train8/weights/best.pt')  # load an official model

# Predict with the model
results = model("lemna_test.png", show=False, conf=0.3, iou=0.1)  # predict on an image
# Show the results
for r in results:
    im_array = r.plot(labels=False, boxes=False, probs=False, masks=True)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    #im.save('results.jpg')  # save image