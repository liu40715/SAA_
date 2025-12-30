import cv2
import numpy as np

# 用來存儲點擊座標的列表
points = []

def draw_polygon(event, x, y, flags, param):
    global points
    
    # 點擊滑鼠左鍵記錄座標
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"點擊座標: [{x}, {y}]")
        
        # 在畫面上畫個小紅點標記一下
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select ROI", img_display)

def interactive_roi(image_path):
    global img_display, points
    
    # 讀取圖片
    img = cv2.imread(image_path)
    # # --- 縮放比例 ---
    # scale_percent = 0.5  # 0.5 代表縮小成 50%
    # # 計算新的寬高 (記得轉成整數 int)
    # width = int(img.shape[1] * scale_percent)
    # height = int(img.shape[0] * scale_percent)
    # dim = (width, height)
    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if img is None: return
    
    # 備份一份用來顯示
    img_display = img.copy()

    print("--- 操作說明 ---")
    print("1. 請用滑鼠左鍵點擊多邊形的角點 (順時針或逆時針)")
    print("2. 點完後，按任意鍵盤按鍵 (如 Space) 開始裁切")
    print("----------------")

    cv2.imshow("Select ROI", img_display)
    cv2.setMouseCallback("Select ROI", draw_polygon)

    # 等待使用者點擊並按鍵
    cv2.waitKey(0)
    
    # 如果有點擊座標
    if len(points) > 2:
        # 轉換座標格式
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 建立遮罩
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # 填滿多邊形
        cv2.fillPoly(mask, [pts], 255)
        
        # 去背
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # 顯示結果
        cv2.imshow("Cropped Result", cv2.resize(result, (0, 0), fx=0.5, fy=0.5))
        cv2.imwrite("Cropped.jpg",result)
        
        # 印出剛剛選取的座標陣列，方便您複製到正式程式碼中
        print("\n--- 請複製以下座標到正式程式碼 ---")
        print(f"pts = np.array({points}, np.int32)")
        print("----------------------------------")
        
        cv2.waitKey(0)
    else:
        print("錯誤：至少需要點選 3 個點才能構成多邊形")

    cv2.destroyAllWindows()

# 執行
interactive_roi("2.jpg")