from PIL import Image
import cv2

# Read an image from filesystem
image = cv2.imread(r"images\image1.jpg")

# optimize using PIL's optimize function
image.save(r'images\PIL_optimizer.jpg',quality=20,optimize=True)