"""Select the full reference object in an image; print its pixel height for calibration CSV."""
import argparse
import cv2
p=argparse.ArgumentParser();p.add_argument('image');a=p.parse_args()
im=cv2.imread(a.image)
if im is None:raise SystemExit('Cannot read image')
x,y,w,h=cv2.selectROI('Select full reference object; Enter confirms',im,False,False)
cv2.destroyAllWindows()
print(f'width={im.shape[1]}, height={im.shape[0]}, pixel_height={h}, box={x,y,w,h}')
