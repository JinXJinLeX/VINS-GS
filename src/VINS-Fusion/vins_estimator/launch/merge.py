import cv2
import os

# 设置视频编码器、帧率、分辨率
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

# 图像文件夹路径
folder = '/home/seu/xjl_work_space/VINS-Fusion/src/VINS-Fusion/vins_estimator/launch/images/'
images = [img for img in os.listdir(folder) if img.endswith(".jpg")]
images.sort()  # 确保文件名按顺序排列

for image in images:
    img = cv2.imread(os.path.join(folder, image))
    if img is not None:
        out.write(img)  # 写入帧
    else:
        print(f"Error loading image {image}")

# 释放VideoWriter对象
out.release()
cv2.destroyAllWindows()



