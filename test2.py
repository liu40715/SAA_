import cv2
import numpy as np
import math

def calculate_skew_angle(image_path):
    # 1. 讀取影像
    img = cv2.imread(image_path)
    if img is None:
        print("無法讀取圖片")
        return

    # 備份一份用來畫圖
    output_img = img.copy()
    
    # 2. 前處理 (Preprocessing)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # [重要] 高斯模糊：金屬表面紋理多，模糊可以減少不必要的雜訊邊緣
    # kernel size (5, 5) 可以根據實際雜訊程度調整
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny 邊緣檢測
    # 這裡的閾值 (50, 150) 需要根據現場光線調整
    # 數字越小，偵測到的邊緣越多；數字越大，只留強烈邊緣
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # [建議] 設定 ROI (感興趣區域)
    # 為了避免偵測到旁邊的直立貨架，我們只看畫面中間部分
    # 這裡假設 FOUP 大致在畫面中間，您可以根據實際情況調整座標
    height, width = edges.shape
    mask = np.zeros_like(edges)
    # 這裡簡單定義一個矩形 ROI：略過左右兩側各 15% 的區域，略過上下 10%
    roi_x1, roi_x2 = int(width * 0.15), int(width * 0.85)
    roi_y1, roi_y2 = int(height * 0.10), int(height * 0.90)
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
    
    # 將 ROI 應用到邊緣圖上
    masked_edges = cv2.bitwise_and(edges, mask)

    # 4. 霍夫變換 (Probabilistic Hough Line Transform)
    # rho: 距離解析度 (1 pixel)
    # theta: 角度解析度 (1度 = np.pi/180)
    # threshold: 只有超過 100 個投票點的線才被認定
    # minLineLength: 線段最小長度 (太短的雜訊不要)
    # maxLineGap: 線段中間允許的斷裂長度
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, 
                            minLineLength=100, maxLineGap=20)

    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 5. 計算角度
            # 使用 arctan2 計算斜率角度
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            
            # 6. 角度過濾 (關鍵步驟)
            # 您的目標是水平放置的 FOUP，所以我們只關心接近 0 度或 180 度的線
            # 這裡設定過濾條件：只保留水平方向 +/- 10 度以內的線
            # 這樣可以濾掉垂直的貨架柱子
            if abs(angle_deg) < 10: 
                angles.append(angle_deg)
                
                # 在圖上畫出這條線 (綠色) - 這是被採納的線
                cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # 畫出被過濾掉的線 (紅色) - 方便 Debug
                cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 7. 統計結果
    if angles:
        # 取平均值作為最終判定的歪斜角度
        avg_angle = np.mean(angles)
        print(f"偵測到 {len(angles)} 條水平特徵線")
        print(f"平均歪斜角度: {avg_angle:.2f} 度")
        
        # 顯示結果文字
        cv2.putText(output_img, f"Skew: {avg_angle:.2f} deg", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        print("未偵測到足夠的水平線特徵")

    # 顯示圖片 (縮小一點以免螢幕放不下)
    display_img = cv2.resize(output_img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Result", display_img)
    cv2.imshow("Edges", cv2.resize(masked_edges, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替換成您的檔名
calculate_skew_angle("2.jpg")