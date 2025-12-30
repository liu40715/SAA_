import cv2

# --- 設定圖片路徑 ---
IMAGE_PATH = '4.jpg'  # 請修改這裡

def click_event(event, x, y, flags, param):
    """
    滑鼠事件的回呼函式 (Callback Function)
    """
    # 偵測是否點擊了 "滑鼠左鍵" (EVENT_LBUTTONDOWN)
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # 1. 在終端機印出座標
        print(f"📍 點擊位置 - X: {x}, Y: {y}")

        # 2. 在圖片上顯示座標文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"({x}, {y})"
        
        # 在點擊的地方畫一個小圓點
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
        # 在點擊的地方寫上座標文字 (黃色字體)
        cv2.putText(img, text, (x + 10, y - 10), font, 0.7, (0, 255, 255), 2)
        
        # 更新顯示圖片
        cv2.imshow('Image Window', img)

# --- 主程式 ---
if __name__ == "__main__":
    # 讀取圖片
    img = cv2.imread(IMAGE_PATH)

    # 檢查圖片是否讀取成功
    if img is None:
        print(f"❌ 錯誤：找不到圖片，請檢查路徑：{IMAGE_PATH}")
    else:
        print("✅ 程式已啟動！請在圖片上點擊滑鼠左鍵...")
        print("ℹ️  按下 'q' 鍵或 'Esc' 鍵可離開程式")

        # 建立視窗
        cv2.imshow('Image Window', img)

        # 設定滑鼠回呼函式 (將視窗與 click_event 函式綁定)
        cv2.setMouseCallback('Image Window', click_event)

        # 等待按鍵，按下 'q' 或 Esc (27) 退出
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        # 關閉所有視窗
        cv2.destroyAllWindows()