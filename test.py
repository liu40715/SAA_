import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import yaml 
import tempfile
import shutil
from supervision.metrics import MeanAveragePrecision, Precision, Recall
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QSizePolicy,
    QGridLayout, QScrollArea, QProgressDialog, QSlider,QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import Qt, QTimer

# ---------------- Resource Path ----------------
def resource_path(filename):
    """
    取得 icon.ico 路徑，假設它在程式上層資料夾
    """
    # __file__ 是當前 .py 檔案
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # 上層資料夾
    return os.path.join(parent_dir, filename)


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAA 模型測試")
        self.setWindowIcon(QIcon(resource_path("icon.ico")))
        self.resize(1400, 900)

        self.device = 'cuda:0' if len(sys.argv) > 1 and sys.argv[1].lower() == 'cpu' else 'cuda:0' 
        
        if self.device == 'cuda:0':
            try:
                # 簡單檢查 CUDA 是否可用
                import torch
                if not torch.cuda.is_available():
                    self.device = 'cpu'
                    print("CUDA 不可用，切換到 CPU 模式。")
                else:
                    print(f"使用 CUDA 設備: {torch.cuda.get_device_name(0)}")
            except:
                self.device = 'cpu'
                print("未安裝 PyTorch 或 CUDA 檢查失敗，切換到 CPU 模式。")
        else:
            self.device = 'cpu'
            print("使用 CPU 模式。")

        # 建立 central widget
        central = QWidget()
        self.setCentralWidget(central)

        # 掃描 models
        self.models = self.scan_models("projects")

        # 模型選擇區
        self.model_box1 = QComboBox()
        self.model_box2 = QComboBox()
        if self.models:
            self.model_box1.addItems(list(self.models.keys()))
            self.model_box2.addItems(["(不選)"] + list(self.models.keys()))
        else:
            self.model_box1.addItems(["<未找到模型>"])
            self.model_box2.addItems(["(不選)"])

        label1 = QLabel("模型1:")
        label1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label2 = QLabel("模型2:")
        label2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        top_layout.addWidget(label1)
        top_layout.addWidget(self.model_box1)
        top_layout.addWidget(label2)
        top_layout.addWidget(self.model_box2)

        # 操作按鈕
        self.btn_images = QPushButton("選擇圖片")
        self.btn_video = QPushButton("選擇影片")
        self.btn_check_labels = QPushButton("測試標註資料夾")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_images)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_check_labels)

        # ScrollArea 顯示區
        self.grid = QGridLayout()
        self.scroll_widget = QWidget()
        self.scroll_widget.setLayout(self.grid)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.scroll_widget)
        self.scroll.setWidgetResizable(True)

        # 影片進度條與暫停按鈕
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)  # 初始禁用
        self.btn_pause = QPushButton("暫停")
        self.btn_pause.setVisible(False)  # 初始隱藏
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.video_slider.sliderReleased.connect(self.slider_seek)

        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.video_slider)
        slider_layout.addWidget(self.btn_pause)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(self.scroll)
        layout.addLayout(slider_layout)
        central.setLayout(layout)

        # 綁定事件
        self.btn_images.clicked.connect(self.load_images)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_check_labels.clicked.connect(self.test_labeled_folder)

        # 初始化變數
        self.image_files = []
        self.loaded_models = {}
        self.labels_row1 = []
        self.labels_row2 = []
        self.metrics_label = None
        
        # 影片相關
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_video_mode = False
        self.video_has_model2 = False
        self.model1 = None
        self.model2 = None
        

        # --------- 建立 annotator ---------
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    # ===== 模型掃描 =====
    def scan_models(self, base_dir="projects"):
        model_dict = {}
        for project in os.listdir(base_dir):
            project_path = os.path.join(base_dir, project)
            runs_path = os.path.join(project_path, "runs")
            if not os.path.isdir(runs_path):
                continue
            for version in os.listdir(runs_path):
                version_path = os.path.join(runs_path, version, "weights", "best.pt")
                if os.path.exists(version_path):
                    key = f"{project} / {version}"
                    model_dict[key] = version_path
        return model_dict
    
    def _clear_previous_results(self):
        """清除網格佈局中舊的結果。"""
        if self.metrics_label:
            self.grid.removeWidget(self.metrics_label)
            self.metrics_label.deleteLater()
            self.metrics_label = None
            
        for label in self.labels_row1:
            self.grid.removeWidget(label)
            label.deleteLater()
        for label in self.labels_row2:
            self.grid.removeWidget(label)
            label.deleteLater()
        self.labels_row1.clear()
        self.labels_row2.clear()

    # =========================================================
    # 新增輔助方法：自動生成 YAML
    # =========================================================
    
    def _get_class_ids_from_labels(self, labels_folder):
        """掃描所有標籤檔案，找出使用的所有類別 ID。"""
        class_ids = set()
        for label_file in os.listdir(labels_folder):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_folder, label_file)
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                # YOLO 格式的第一個數字是類別 ID
                                class_ids.add(int(parts[0]))
                except Exception:
                    continue
        return class_ids

    def _create_temp_yaml(self, root_dir, class_ids):
        """
        在臨時目錄中創建一個 data.yaml 檔案，並返回路徑。
        
        Args:
            root_dir (str): 資料夾的根路徑 (包含 images/ 和 labels/)。
            class_ids (set): 標籤文件中找到的所有類別 ID 集合。
            
        Returns:
            str: 臨時 data.yaml 檔案的路徑。
        """
        max_class_id = max(class_ids) if class_ids else -1
        num_classes = max_class_id + 1
        
        # 創建類別名稱列表
        class_names = [f'class_{i}' for i in range(num_classes)]
        
        data_yaml_content = {
            # train 和 val 都指向根目錄，讓 val() 函式能找到 images/ 和 labels/
            'path': root_dir,
            'train': 'images', # 實際上 val 模式下，會查找 val 或 test
            'val': 'images',
            'test': 'images',
            'nc': num_classes,
            'names': class_names
        }

        # 在臨時目錄中創建檔案
        temp_dir = tempfile.mkdtemp()
        yaml_path = os.path.join(temp_dir, "temp_data.yaml")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f)
            
        return yaml_path, temp_dir
    
    # =========================================================
    # 核心功能: 測試資料夾 (使用 model.val())
    # =========================================================

    def test_labeled_folder(self):
        """
        測試已標註的資料夾，使用 model.val() 計算總體指標，並自動生成 YAML。
        """
        folder = QFileDialog.getExistingDirectory(self, "選擇已標註的資料夾 (包含 images/ 和 labels/ 子資料夾)")
        if not folder:
            return

        images_folder = os.path.join(folder, "images")
        labels_folder = os.path.join(folder, "labels")
        if not os.path.isdir(images_folder) or not os.path.isdir(labels_folder):
            QMessageBox.warning(self, "資料夾檢查", "資料夾需要有 **images** 和 **labels** 子資料夾！")
            return

        # 清除舊的結果顯示
        self._clear_previous_results()
        
        # 檔案過濾和檢查
        image_exts = (".jpg", ".jpeg", ".png")
        all_image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(image_exts)])
        label_basenames = set(
            os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.lower().endswith(".txt")
        )
        image_files = [f for f in all_image_files if os.path.splitext(f)[0] in label_basenames]

        if not image_files:
            QMessageBox.warning(self, "資料檢查", "沒有對應標註的圖片可以測試！")
            return
            
        # 載入模型
        # ... (模型載入邏輯與之前相同) ...
        has_model2 = self.model_box2.currentText() in self.models
        name1 = self.model_box1.currentText()
        name2 = self.model_box2.currentText() if has_model2 else None
        
        self.model1, err1 = self._load_model_by_name(name1)
        if err1:
            QMessageBox.warning(self, "模型載入失敗", err1)
            return
            
        self.model2 = None
        if has_model2:
            self.model2, err2 = self._load_model_by_name(name2)
            if err2:
                QMessageBox.warning(self, "模型載入失敗", err2)
                return

        # =========================================================
        # 步驟 1: 自動生成 YAML 檔案
        # =========================================================
        temp_yaml_path = None
        temp_dir = None
        
        try:
            class_ids = self._get_class_ids_from_labels(labels_folder)
            temp_yaml_path, temp_dir = self._create_temp_yaml(folder, class_ids)
            
            progress = QProgressDialog("計算指標 (val)...", "", 0, 0, self) 
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("計算總體指標中...")
            progress.show()
            QApplication.processEvents()
            
            # --- 執行 model.val() ---
            results1 = self.model1.val(
                data=temp_yaml_path, # 使用臨時生成的 YAML 檔案
                conf=0.25, 
                iou=0.5,     
                device=self.device,
                verbose=False
            )
            
            results2 = None
            if has_model2 and self.model2:
                results2 = self.model2.val(
                    data=temp_yaml_path, # 使用臨時生成的 YAML 檔案
                    conf=0.25, 
                    iou=0.5,
                    device=self.device,
                    verbose=False
                )
            
            progress.close()

            # 提取並顯示指標
            mAP50_95_1 = float(results1.box.map)
            mAP50_1 = float(results1.box.map50)
            prec1 = float(results1.box.p)
            rec1 = float(results1.box.r)

            metrics_text = f"🥇 **模型1** ({name1}) 評估 (來自 **model.val()**):\n"
            metrics_text += f"**mAP50‑95**: {mAP50_95_1:.4f}, **mAP50**: {mAP50_1:.4f}\n"
            metrics_text += f"**Precision**: {prec1:.4f}, **Recall**: {rec1:.4f}\n"

            if results2:
                mAP50_95_2 = float(results2.box.map)
                mAP50_2 = float(results2.box.map50)
                prec2 = float(results2.box.p)
                rec2 = float(results2.box.r)
                
                metrics_text += f"\n🥈 **模型2** ({name2}) 評估 (來自 **model.val()**):\n"
                metrics_text += f"**mAP50‑95**: {mAP50_95_2:.4f}, **mAP50**: {mAP50_2:.4f}\n"
                metrics_text += f"**Precision**: {prec2:.4f}, **Recall**: {rec2:.4f}\n"

            
            self.metrics_label = QLabel(metrics_text)
            self.grid.addWidget(self.metrics_label, 0, 0, 1, 2)
            
            # =========================================================
            # 步驟 2 & 3: 推理、儲存結果並繪圖 (與之前邏輯相同)
            # =========================================================
            
            n_cols = 2 if has_model2 else 1
            # 獲取視窗寬度用於縮放圖片
            col_width = (self.scroll.viewport().width() - 20) // max(n_cols, 1)

            progress.setWindowTitle("🖼️ 標記圖片")
            progress.setRange(0, len(image_files))

            all_det1 = []
            all_det2 = []
            
            for idx, img_file in enumerate(image_files):
                progress.setValue(idx)
                QApplication.processEvents()

                img_path = os.path.join(images_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 模型1 單圖推理
                results1_inference = self.model1(img, conf=0.25, device=self.device, verbose=False)[0]
                det1 = sv.Detections.from_ultralytics(results1_inference)
                all_det1.append(det1)

                # 模型2 單圖推理
                if has_model2 and self.model2:
                    results2_inference = self.model2(img, conf=0.25, device=self.device, verbose=False)[0]
                    det2 = sv.Detections.from_ultralytics(results2_inference)
                    all_det2.append(det2)

            progress.setWindowTitle("繪製圖片中...")
            for idx, img_file in enumerate(image_files):
                progress.setValue(idx)
                QApplication.processEvents()

                img_path = os.path.join(images_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                det1 = all_det1[idx]
                self._annotate_and_display_image(self.model1, img, det1, 0, idx + 1, col_width)

                if has_model2 and self.model2:
                    det2 = all_det2[idx]
                    self._annotate_and_display_image(self.model2, img, det2, 1, idx + 1, col_width)
            
            progress.setValue(len(image_files))
            progress.close()
            
        except Exception as e:
            if progress.isVisible():
                 progress.close()
            QMessageBox.critical(self, "錯誤", f"在處理或驗證過程中發生錯誤: {str(e)}")
            
        finally:
            # 確保刪除臨時資料夾及其內容
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg)")
        if not files:
            return

        self.is_video_mode = False
        self.image_files = files
        self.video_slider.setEnabled(False)
        self.btn_pause.setVisible(False)

        has_model2 = self.model_box2.currentText() in self.models
        # -------- 載入模型 --------
        try:
            # 模型1
            model1_name = self.model_box1.currentText()
            if model1_name not in self.loaded_models:
                self.loaded_models[model1_name] = YOLO(self.models[model1_name]).to(self.device)
            self.model1 = self.loaded_models[model1_name]

            # 模型2
            self.model2 = None
            if has_model2:
                model2_name = self.model_box2.currentText()
                if model2_name not in self.loaded_models:
                    self.loaded_models[model2_name] = YOLO(self.models[model2_name]).to(self.device)
                self.model2 = self.loaded_models[model2_name]

        except Exception as e:
            QMessageBox.warning(self, "模型載入失敗", f"載入模型時發生錯誤:\n{str(e)}")
            return

        progress = QProgressDialog("載入圖片中...", "", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setWindowTitle("載入圖片")
        progress.show()

        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.labels_row1.clear()
        self.labels_row2.clear()

        n_cols = 2 if has_model2 else 1
        col_width = self.scroll.viewport().width() // max(n_cols, 1) - 10

        # -------- 處理圖片 --------
        for idx, img_path in enumerate(files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 模型1
            results1 = self.model1(img, conf=0.25, device=self.device)[0]
            det1 = sv.Detections.from_ultralytics(results1)
            labels1 = [f"{self.model1.model.names[cid]} {conf:.2f}" for cid, conf in zip(det1.class_id, det1.confidence)]
            ann1 = self.box_annotator.annotate(scene=img.copy(), detections=det1)
            ann1 = self.label_annotator.annotate(scene=ann1, detections=det1, labels=labels1)
            pix1 = self.cv2_to_qpixmap(ann1)
            label1 = QLabel()
            label1.setPixmap(pix1.scaledToWidth(col_width, Qt.SmoothTransformation))
            self.grid.addWidget(label1, idx, 0)
            self.labels_row1.append(label1)

            # 模型2
            if has_model2 and self.model2:
                results2 = self.model2(img, conf=0.25, device=self.device)[0]
                det2 = sv.Detections.from_ultralytics(results2)
                labels2 = [f"{self.model2.model.names[cid]} {conf:.2f}" for cid, conf in zip(det2.class_id, det2.confidence)]
                ann2 = self.box_annotator.annotate(scene=img.copy(), detections=det2)
                ann2 = self.label_annotator.annotate(scene=ann2, detections=det2, labels=labels2)
                pix2 = self.cv2_to_qpixmap(ann2)
                label2 = QLabel()
                label2.setPixmap(pix2.scaledToWidth(col_width, Qt.SmoothTransformation))
                self.grid.addWidget(label2, idx, 1)
                self.labels_row2.append(label2)

            progress.setValue(idx + 1)
            QApplication.processEvents()

        progress.close()

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", "", "Videos (*.mp4 *.avi *.mov)")
        if not path:
            return

        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.cap = cap
        self.is_video_mode = True
        self.video_has_model2 = self.model_box2.currentText() in self.models

        # -------- 載入模型 --------
        try:
            model1_name = self.model_box1.currentText()
            if model1_name not in self.loaded_models:
                self.loaded_models[model1_name] = YOLO(self.models[model1_name]).to(self.device)
            self.model1 = self.loaded_models[model1_name]

            self.model2 = None
            if self.video_has_model2:
                model2_name = self.model_box2.currentText()
                if model2_name not in self.loaded_models:
                    self.loaded_models[model2_name] = YOLO(self.models[model2_name]).to(self.device)
                self.model2 = self.loaded_models[model2_name]

        except Exception as e:
            QMessageBox.warning(self, "模型載入失敗", f"載入模型時發生錯誤:\n{str(e)}")
            return

        progress = QProgressDialog("載入影片中...", "", 0, frame_count, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setWindowTitle("載入影片")
        progress.show()

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            progress.setValue(i + 1)
            QApplication.processEvents()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        progress.close()  # ❌ 讀完影片幀後關閉

        # 清空 grid
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.labels_row1.clear()
        self.labels_row2.clear()

        # 建立顯示 label
        label1 = QLabel()
        label1.setAlignment(Qt.AlignCenter)
        self.grid.addWidget(label1, 0, 0)
        self.labels_row1.append(label1)

        if self.video_has_model2:
            label2 = QLabel()
            label2.setAlignment(Qt.AlignCenter)
            self.grid.addWidget(label2, 0, 1)
            self.labels_row2.append(label2)

        self.btn_pause.setVisible(True)
        self.video_slider.setMaximum(frame_count - 1)
        self.video_slider.setEnabled(True)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = int(1000 / fps)
        self.timer.start(interval)


    def next_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # 更新滑桿
        self.video_slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

        col_width = self.scroll.viewport().width() // (2 if self.video_has_model2 else 1) - 10

        # 模型1
        results1 = self.model1(frame, conf=0.25, device=self.device)[0]
        det1 = sv.Detections.from_ultralytics(results1)
        labels1 = [
            f"{self.model1.model.names[class_id]} {conf:.2f}"
            for class_id, conf in zip(det1.class_id, det1.confidence)
        ]
        ann1 = self.box_annotator.annotate(scene=frame.copy(), detections=det1)
        ann1 = self.label_annotator.annotate(scene=ann1, detections=det1, labels=labels1)
        pix1 = self.cv2_to_qpixmap(ann1)
        self.labels_row1[0].setPixmap(pix1.scaledToWidth(col_width, Qt.SmoothTransformation))

        # 模型2
        if self.video_has_model2 and self.model2:
            results2 = self.model2(frame, conf=0.25, device=self.device)[0]
            det2 = sv.Detections.from_ultralytics(results2)
            labels2 = [
                f"{self.model2.model.names[class_id]} {conf:.2f}"
                for class_id, conf in zip(det2.class_id, det2.confidence)
            ]
            ann2 = self.box_annotator.annotate(scene=frame.copy(), detections=det2)
            ann2 = self.label_annotator.annotate(scene=ann2, detections=det2, labels=labels2)
            pix2 = self.cv2_to_qpixmap(ann2)
            self.labels_row2[0].setPixmap(pix2.scaledToWidth(col_width, Qt.SmoothTransformation))

    # ===== 暫停/播放 =====
    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("播放")
        else:
            self.timer.start()
            self.btn_pause.setText("暫停")

    # ===== 拖曳滑桿跳轉 =====
    def slider_seek(self):
        if not self.cap:
            return
        pos = self.video_slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = self.cap.read()
        if ret:
            col_width = self.scroll.viewport().width() // (2 if self.video_has_model2 else 1) - 10
            # 模型1
            results1 = self.model1(frame, conf=0.25, device=self.device)[0]
            det1 = sv.Detections.from_ultralytics(results1)
            labels1 = [
                f"{self.model1.model.names[class_id]} {conf:.2f}"
                for class_id, conf in zip(det1.class_id, det1.confidence)
            ]
            ann1 = self.box_annotator.annotate(scene=frame.copy(), detections=det1)
            ann1 = self.label_annotator.annotate(scene=ann1, detections=det1, labels=labels1)
            pix1 = self.cv2_to_qpixmap(ann1)
            self.labels_row1[0].setPixmap(pix1.scaledToWidth(col_width, Qt.SmoothTransformation))

            # 模型2
            if self.video_has_model2 and self.model2:
                results2 = self.model2(frame, conf=0.25, device=self.device)[0]
                det2 = sv.Detections.from_ultralytics(results2)
                labels2 = [
                    f"{self.model2.model.names[class_id]} {conf:.2f}"
                    for class_id, conf in zip(det2.class_id, det2.confidence)
                ]
                ann2 = self.box_annotator.annotate(scene=frame.copy(), detections=det2)
                ann2 = self.label_annotator.annotate(scene=ann2, detections=det2, labels=labels2)
                pix2 = self.cv2_to_qpixmap(ann2)
                self.labels_row2[0].setPixmap(pix2.scaledToWidth(col_width, Qt.SmoothTransformation))

    # ===== 共用函數 =====
    def _load_model_by_name(self, name):
        """
        [佔位符] 根據名稱載入 YOLO 模型。
        
        Args:
            name (str): 模型的名稱 (例如: "model_a.pt")。
            
        Returns:
            tuple[YOLO, str]: (YOLO 模型實例, 錯誤訊息)
        """
        if name in self.models:
            try:
                model = YOLO(self.models[name]).to(device=self.device)
                return model, None
            except Exception as e:
                return None, f"YOLO 模型載入失敗: {e}"
        return None, f"找不到模型: {name}"

    def cv2_to_qpixmap(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _annotate_and_display_image(self, model, img, det, col_index, row_index, col_width):
        """
        標註圖片並顯示在 Grid Layout 中。
        """
        ann_img = img.copy() 
        
        # 創建標籤 (包含類別名稱和信心度)
        labels = [
            f"{model.model.names.get(cid, 'Unknown')} {conf:.2f}" 
            for cid, conf in zip(det.class_id, det.confidence)
        ]
        
        # 使用 Supervision 標註
        ann_img = self.box_annotator.annotate(scene=ann_img, detections=det)
        ann_img = self.label_annotator.annotate(scene=ann_img, detections=det, labels=labels)
        
        # 轉換為 QPixmap 並設置大小
        pix = self.cv2_to_qpixmap(ann_img)
        label = QLabel()
        label.setPixmap(pix.scaledToWidth(col_width, Qt.SmoothTransformation))
        
        # 添加到網格佈局
        self.grid.addWidget(label, row_index, col_index)
        
        # 記錄 QLabel 方便之後清理 (如果需要)
        if col_index == 0:
            self.labels_row1.append(label)
        elif col_index == 1:
            self.labels_row2.append(label)

    def resizeEvent(self, event):
        if self.is_video_mode:
            col_width = self.scroll.viewport().width() // (2 if self.video_has_model2 else 1) - 10
            self.labels_row1[0].setFixedWidth(col_width)
            if self.video_has_model2:
                self.labels_row2[0].setFixedWidth(col_width)
        else:
            self.display_images()
        super().resizeEvent(event)

    def display_images(self):
        has_model2 = self.model_box2.currentText() in self.models
        n_cols = 2 if has_model2 else 1
        col_width = self.scroll.viewport().width() // max(n_cols, 1) - 10

        for idx, img_path in enumerate(self.image_files):
            img = cv2.imread(img_path)
            pixmap = self.cv2_to_qpixmap(img)

            if idx < len(self.labels_row1):
                self.labels_row1[idx].setPixmap(pixmap.scaledToWidth(col_width, Qt.SmoothTransformation))
            if has_model2 and idx < len(self.labels_row2):
                self.labels_row2[idx].setPixmap(pixmap.scaledToWidth(col_width, Qt.SmoothTransformation))
    
    # ===================== closeEvent =====================
    def closeEvent(self, event):
        # 停止 timer
        if self.timer.isActive():
            self.timer.stop()
        # 釋放影片資源
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.showMaximized()
    sys.exit(app.exec())
