import sys
import os
import cv2
from ultralytics import YOLO


model = YOLO(r'C:\Users\Danny.Liu\Desktop\yolo_test\best.pt')
model.to('cuda')
# 2. 讀取圖片 (保留原圖)
img_path = r'C:\Users\Danny.Liu\Desktop\yolo_test\4.jpg'
ori_img = cv2.imread(img_path)
img = ori_img.copy()
results = model(img)
# 4. 遍歷每一個偵測到的物件
i = 0
for box in results[0].boxes:
    # --- A. 取出座標 ---
    # xyxy[0] 代表 [x1, y1, x2, y2]
    # 記得要轉成 int 整數才能畫圖
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop_img = ori_img[y1:y2, x1:x2]
    save_path = os.path.join(r"./", f"crop_{i}.jpg")
    i+=1
    # 只有當切出來的圖片不為空時才存檔
    if crop_img.size > 0:
        cv2.imwrite(save_path, crop_img)

    # --- B. 取出信心度與類別 (選用) ---
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = f"{model.names[cls_id]} {conf:.2f}"

    # --- C. 畫框 (顏色: BGR, 這裡是綠色, 粗細: 2) ---
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- D. 寫字 (選用) ---
    # 參數: 圖片, 文字, 位置, 字型, 大小, 顏色, 粗細
    # cv2.putText(img, label, (x1, y1 - 10), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# --- 縮放比例 ---
scale_percent = 0.5  # 0.5 代表縮小成 50%
# 計算新的寬高 (記得轉成整數 int)
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)

# 執行縮放
resized_frame = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# 5. 顯示結果
cv2.imshow("Manual Drawing", resized_frame)
cv2.waitKey(0)




# result = results[0]  
# # 4. 讀取並顯示邊界框座標
# # result.boxes.xyxy 是 GPU/CPU 上的 Tensor，用 .tolist() 轉成 Python 列表比較好讀
# for box in result.boxes.xyxy.tolist():
#     x1, y1, x2, y2 = box
    
#     # 為了顯示好看，通常會轉成整數
#     print(f"邊界框: 左上({int(x1)}, {int(y1)}) -> 右下({int(x2)}, {int(y2)})")

# # [補充] 如果你想看這張圖被畫出來的樣子
# result.show()