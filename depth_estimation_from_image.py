import cv2
import numpy as np
import torch
import argparse
import os
import json
import math
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthYOLOEstimator:
    def __init__(self, yolo_model="yolo11n.pt"):
        """
        Khởi tạo bộ ước lượng độ sâu Depth Anything V2 và YOLO
        
        Args:
            yolo_model: Đường dẫn đến model YOLOv11
        """
        # Tải mô hình YOLO
        print(f"Đang tải mô hình YOLOv11: {yolo_model}...")
        try:
            self.yolo_model = YOLO(yolo_model)
            print("Đã tải mô hình YOLOv11 thành công!")
        except Exception as e:
            print(f"Lỗi khi tải mô hình YOLOv11: {e}")
            print("Vui lòng đảm bảo rằng bạn đã cài đặt Ultralytics và tải mô hình YOLOv11.")
            exit(1)
        
        # Tải mô hình Depth Anything V2
        print("Đang tải mô hình Depth Anything V2...")
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
            self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
            
            if torch.cuda.is_available():
                self.depth_model = self.depth_model.to("cuda")
                print("Sử dụng GPU để tăng tốc!")
            else:
                print("Sử dụng CPU (quá trình ước lượng có thể chậm hơn).")
            print("Đã tải mô hình Depth Anything V2 thành công!")
        except Exception as e:
            print(f"Lỗi khi tải mô hình Depth Anything V2: {e}")
            print("Vui lòng đảm bảo rằng bạn đã cài đặt transformers phiên bản mới nhất.")
            exit(1)
        
        # Hằng số chuyển đổi từ độ sâu ước lượng sang khoảng cách thực (mét)
        self.scale_factor = 0.1
        
        # Cài đặt ngưỡng phát hiện cho YOLO
        self.confidence_threshold = 0.25
        
        # Danh sách màu cho các đối tượng khác nhau
        np.random.seed(50)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    def estimate_depth(self, image):
        """
        Ước lượng độ sâu sử dụng mô hình Depth Anything V2
        
        Args:
            image: Ảnh đầu vào (BGR)
            
        Returns:
            depth_map: Bản đồ độ sâu
            colormap: Bản đồ độ sâu dạng màu để hiển thị
        """
        # Chuyển sang RGB cho mô hình
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Tiền xử lý ảnh
        inputs = self.depth_processor(images=rgb_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Dự đoán độ sâu
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Xử lý đầu ra
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Chuẩn hóa bản đồ độ sâu để hiển thị
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        normalized_depth = normalized_depth.astype(np.uint8)
        
        # Tạo bản đồ màu
        colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        
        return depth_map, colormap
    
    def detect_objects(self, image):
        """
        Phát hiện đối tượng trong ảnh sử dụng YOLOv11
        
        Args:
            image: Ảnh đầu vào (BGR)
            
        Returns:
            results: Kết quả phát hiện đối tượng
        """
        # Phát hiện đối tượng
        results = self.yolo_model(image, conf=self.confidence_threshold)
        return results[0]  # Lấy kết quả cho ảnh đầu tiên (và duy nhất)
    
    def measure_distance_to_bbox(self, depth_map, bbox):
        """
        Đo khoảng cách trung bình đến một bounding box
        
        Args:
            depth_map: Bản đồ độ sâu
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            distance: Khoảng cách (mét)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Giới hạn tọa độ bbox trong phạm vi của depth_map
        x1 = max(0, min(x1, depth_map.shape[1] - 1))
        y1 = max(0, min(y1, depth_map.shape[0] - 1))
        x2 = max(0, min(x2, depth_map.shape[1] - 1))
        y2 = max(0, min(y2, depth_map.shape[0] - 1))
        
        # Lấy vùng depth tương ứng với bbox
        depth_region = depth_map[y1:y2, x1:x2]
        
        if depth_region.size == 0:
            return None
        
        # Tính trung bình độ sâu trong bbox
        avg_depth = np.mean(depth_region)
        
        # Depth Anything V2 trả về thang độ sâu tương đối
        # Chuyển đổi sang khoảng cách thực (mét)
        # Lưu ý: Scale factor cần được hiệu chỉnh dựa trên thực nghiệm
        distance = avg_depth * self.scale_factor
        
        return distance
    
    def calculate_3d_position(self, bbox, depth_map, image_center_x, image_center_y, focal_length=1000):
        """
        Tính toán vị trí 3D của một đối tượng dựa trên bbox và bản đồ độ sâu
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_map: Bản đồ độ sâu
            image_center_x: Tọa độ x của tâm ảnh
            image_center_y: Tọa độ y của tâm ảnh
            focal_length: Độ dài tiêu cự của camera (pixel)
            
        Returns:
            x, y, z: Tọa độ 3D của đối tượng (mét)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Tính tọa độ trung tâm của bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Đo độ sâu trung bình tại bbox
        depth = self.measure_distance_to_bbox(depth_map, bbox)
        if depth is None:
            return None, None, None
        
        # Tính tọa độ 3D sử dụng phép chiếu camera
        # Chuyển tọa độ pixel sang tọa độ thế giới
        x = (center_x - image_center_x) * depth / focal_length
        y = (center_y - image_center_y) * depth / focal_length
        z = depth
        
        return x, y, z
    
    def calculate_3d_angles(self, x, y, z):
        """
        Tính các góc trong không gian 3D
        
        Args:
            x, y, z: Tọa độ 3D của đối tượng
            
        Returns:
            azimuth: Góc đứng (độ) - góc hợp với mặt phẳng XZ
            elevation: Góc ngẩng (độ) - góc hợp với mặt phẳng XY
            roll: Góc lăn (độ) - góc xoay quanh trục Z
        """
        if x is None or y is None or z is None:
            return None, None, None
            
        # Tính góc phương vị (azimuth) - góc trong mặt phẳng XZ
        # Đây là góc giữa đường thẳng từ camera đến đối tượng và trục Z dương
        azimuth = math.degrees(math.atan2(x, z))
        
        # Tính góc ngẩng (elevation) - góc với mặt phẳng XZ
        # Là góc giữa đường thẳng từ camera đến đối tượng và mặt phẳng XZ
        distance_xz = math.sqrt(x**2 + z**2)
        elevation = math.degrees(math.atan2(y, distance_xz))
        
        # Góc lăn (roll) - thường không thể xác định chỉ từ tọa độ vị trí
        # Để tính roll cần thông tin thêm về hướng của đối tượng
        roll = 0  # Mặc định là 0
        
        return azimuth, elevation, roll
    
    def calculate_2d_angle(self, bbox, image_center_x):
        """
        Tính góc của trung tâm bbox so với đường thẳng giữa (2D)
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_center_x: Tọa độ x của đường thẳng giữa
            
        Returns:
            angle: Góc (độ) của trung tâm bbox so với đường thẳng giữa
                  Góc dương: bbox nằm bên phải đường thẳng
                  Góc âm: bbox nằm bên trái đường thẳng
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Tính tọa độ trung tâm của bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Tính khoảng cách theo trục x từ trung tâm bbox đến đường thẳng giữa
        dx = bbox_center_x - image_center_x
        
        # Tính khoảng cách theo trục y từ trung tâm bbox đến đỉnh của ảnh
        dy = bbox_center_y
        
        # Tính góc (theo radian)
        angle_rad = math.atan2(dx, dy)  # atan2(x, y) để tính góc so với trục y
        
        # Chuyển đổi từ radian sang độ
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def calibrate_scale_factor(self, depth_map, bbox, known_distance):
        """
        Hiệu chỉnh hệ số tỷ lệ dựa trên khoảng cách thực đã biết
        
        Args:
            depth_map: Bản đồ độ sâu
            bbox: Bounding box của vật thể ở khoảng cách đã biết [x1, y1, x2, y2]
            known_distance: Khoảng cách thực đến vật thể (mét)
            
        Returns:
            Trả về hệ số tỷ lệ mới
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Giới hạn tọa độ bbox trong phạm vi của depth_map
        x1 = max(0, min(x1, depth_map.shape[1] - 1))
        y1 = max(0, min(y1, depth_map.shape[0] - 1))
        x2 = max(0, min(x2, depth_map.shape[1] - 1))
        y2 = max(0, min(y2, depth_map.shape[0] - 1))
        
        # Lấy vùng depth tương ứng với bbox
        depth_region = depth_map[y1:y2, x1:x2]
        
        if depth_region.size == 0:
            return self.scale_factor
        
        # Tính trung bình độ sâu trong bbox
        avg_depth = np.mean(depth_region)
        
        # Đối với Depth Anything, dùng công thức: distance = depth * scale_factor
        # => scale_factor = known_distance / avg_depth
        new_scale = known_distance / avg_depth
        return new_scale
    
    def draw_3d_coordinate_system(self, image, center_x, center_y, axis_length=100):
        """
        Vẽ hệ trục tọa độ 3D lên ảnh
        
        Args:
            image: Ảnh đầu vào
            center_x, center_y: Tọa độ tâm của hệ trục
            axis_length: Độ dài của các trục
            
        Returns:
            image: Ảnh với hệ trục tọa độ 3D
        """
        # Vẽ trục X (màu đỏ)
        cv2.arrowedLine(image, (center_x, center_y), (center_x + axis_length, center_y), (0, 0, 255), 2)
        cv2.putText(image, "X", (center_x + axis_length + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Vẽ trục Y (màu xanh lá)
        cv2.arrowedLine(image, (center_x, center_y), (center_x, center_y - axis_length), (0, 255, 0), 2)
        cv2.putText(image, "Y", (center_x, center_y - axis_length - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ trục Z (màu xanh dương) - trục này hướng ra phía trước camera
        # Để thể hiện trục Z, ta vẽ nó nhỏ dần (hiệu ứng phối cảnh)
        cv2.line(image, (center_x, center_y), (center_x - int(axis_length*0.5), center_y + int(axis_length*0.5)), (255, 0, 0), 2)
        # Vẽ đầu mũi tên cho trục Z
        arrow_points = np.array([[center_x - int(axis_length*0.5), center_y + int(axis_length*0.5)],
                                 [center_x - int(axis_length*0.5) - 5, center_y + int(axis_length*0.5) - 5],
                                 [center_x - int(axis_length*0.5) + 5, center_y + int(axis_length*0.5) + 5]])
        cv2.fillPoly(image, [arrow_points], (255, 0, 0))
        cv2.putText(image, "Z", (center_x - int(axis_length*0.5) - 20, center_y + int(axis_length*0.5) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return image
    
    def process_image(self, image_path, save_path=None, show_result=True, calibrate=False, focal_length=1000):
        """
        Xử lý ảnh để phát hiện đối tượng và ước lượng khoảng cách trong không gian 3D
        
        Args:
            image_path: Đường dẫn đến ảnh đầu vào
            save_path: Đường dẫn để lưu ảnh kết quả (tùy chọn)
            show_result: Hiển thị kết quả
            calibrate: Bật chế độ hiệu chỉnh
            focal_length: Độ dài tiêu cự của camera (pixel)
            
        Returns:
            result_image: Ảnh với thông tin đối tượng và khoảng cách
            detections: Danh sách các phát hiện với khoảng cách
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh từ {image_path}")
            return None, []
        
        # Điều chỉnh kích thước ảnh nếu quá lớn
        max_dim = 1024
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            if h > w:
                scale = max_dim / h
                new_h, new_w = max_dim, int(w * max_dim / h)
            else:
                scale = max_dim / w
                new_h, new_w = int(h * max_dim / w), max_dim
            image = cv2.resize(image, (new_w, new_h))
            print(f"Đã điều chỉnh kích thước ảnh thành {new_w}x{new_h}")
        
        # Cập nhật kích thước h, w sau khi thay đổi kích thước
        h, w = image.shape[:2]
        
        # Tạo một bản sao của ảnh để vẽ lên
        result_image = image.copy()
        
        # Tính toán tọa độ trung tâm ảnh
        center_x = w // 2
        center_y = h // 2
        
        # Vẽ đường thẳng giữa (đường cơ sở 0 độ)
        cv2.line(result_image, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        
        # Bỏ vẽ hệ trục tọa độ 3D tại trung tâm ảnh
        # result_image = self.draw_3d_coordinate_system(result_image, center_x, center_y)
        
        # Phát hiện đối tượng với YOLOv11
        print("Đang phát hiện đối tượng...")
        detections = self.detect_objects(image)
        
        # Tính toán bản đồ độ sâu
        print("Đang tính toán bản đồ độ sâu...")
        depth_map, colormap = self.estimate_depth(image)
        
        # Vẽ đường thẳng giữa trên bản đồ độ sâu cũng để đồng bộ hiển thị
        cv2.line(colormap, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        
        # Danh sách lưu thông tin phát hiện và khoảng cách
        detection_info = []
        
        # Xử lý từng đối tượng được phát hiện
        boxes = detections.boxes
        for i, box in enumerate(boxes):
            # Lấy thông tin bbox và class
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # Lấy nhãn và màu cho class
            label = detections.names[class_id]
            color = self.colors[class_id % len(self.colors)].tolist()
            
            # Tính toán góc 2D
            angle_2d = self.calculate_2d_angle(bbox, center_x)
            
            # Tính toán vị trí 3D của đối tượng
            x_3d, y_3d, z_3d = self.calculate_3d_position(bbox, depth_map, center_x, center_y, focal_length)
            
            # Tính các góc 3D
            azimuth, elevation, roll = None, None, None
            if x_3d is not None and y_3d is not None and z_3d is not None:
                azimuth, elevation, roll = self.calculate_3d_angles(x_3d, y_3d, z_3d)
            
            # Vẽ bbox và thông tin
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Tính và vẽ trung tâm của bbox
            center_x_bbox = (x1 + x2) // 2
            center_y_bbox = (y1 + y2) // 2
            cv2.circle(result_image, (center_x_bbox, center_y_bbox), 5, color, -1)
            
            # Vẽ đường từ trung tâm bbox đến đường thẳng giữa
            cv2.line(result_image, (center_x_bbox, center_y_bbox), (center_x, center_y_bbox), (0, 255, 255), 2)
            
            # Tạo text hiển thị với góc 2D và 3D
            text_lines = []
            text_lines.append(f"{label} ({conf:.2f})")
            
            if x_3d is not None:
                distance_text = f"Dist: {z_3d:.2f}m"
                text_lines.append(distance_text)
                
                # Bỏ hiển thị tọa độ 3D
                # pos_text = f"Pos(m): X:{x_3d:.2f}, Y:{y_3d:.2f}, Z:{z_3d:.2f}"
                # text_lines.append(pos_text)
            
            if azimuth is not None:
                # Chỉ hiển thị góc 2D, không hiển thị các góc 3D phức tạp
                angle_text = f"Angle: {angle_2d:.1f}°"
                text_lines.append(angle_text)
                
                # Bỏ hiển thị góc azimuth, elevation
                # angle_text = f"Az:{azimuth:.1f}°, El:{elevation:.1f}°"
                # text_lines.append(angle_text)
            
            # Vẽ các dòng text
            text_box_height = len(text_lines) * 20
            # Tính toán chiều rộng cần thiết cho text box
            max_text_width = 0
            for line in text_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                max_text_width = max(max_text_width, text_size[0])
            
            # Thêm padding
            box_width = max_text_width + 10
            
            # Vẽ hộp với kích thước phù hợp
            cv2.rectangle(result_image, (x1, y1 - text_box_height - 5), (x1 + box_width, y1), color, -1)
            
            for i, line in enumerate(text_lines):
                cv2.putText(result_image, line, (x1 + 5, y1 - text_box_height + i*20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Lưu thông tin phát hiện
            detection_info.append({
                'label': label,
                'confidence': conf,
                'bbox': bbox.tolist(),
                'angle_2d': angle_2d,
                'position_3d': (x_3d, y_3d, z_3d) if x_3d is not None else None,
                # 'distance': z_3d,
                'angles_3d': {
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'roll': roll
                } if azimuth is not None else None
            })
            
            # In thông tin vào console
            print(f"\nĐối tượng {i+1}: {label}")
            print(f"  Độ tin cậy: {conf:.2f}")
            if x_3d is not None:
                print(f"  Vị trí 3D (m): X: {x_3d:.2f}, Y: {y_3d:.2f}, Z: {z_3d:.2f}")
            if azimuth is not None:
                print(f"  Góc 3D: Azimuth: {azimuth:.2f}°, Elevation: {elevation:.2f}°")
            print(f"  Góc 2D: {angle_2d:.2f}°")
        
        # Hiệu chỉnh hệ số tỷ lệ nếu cần
        if calibrate and boxes.shape[0] > 0:
            print("\nChế độ hiệu chỉnh:")
            print("Các đối tượng được phát hiện:")
            for i, info in enumerate(detection_info):
                print(f"{i+1}. {info['label']}")
            
            try:
                choice = int(input("Chọn đối tượng để hiệu chỉnh (nhập số): ")) - 1
                if 0 <= choice < len(detection_info):
                    known_distance = float(input("Nhập khoảng cách thực đến đối tượng (mét): "))
                    new_scale = self.calibrate_scale_factor(depth_map, detection_info[choice]['bbox'], known_distance)
                    self.scale_factor = new_scale
                    print(f"Đã hiệu chỉnh hệ số tỷ lệ: {new_scale}")
                    
                    # Xử lý lại ảnh với hệ số tỷ lệ mới
                    return self.process_image(image_path, save_path, show_result, False, focal_length)
                else:
                    print("Lựa chọn không hợp lệ.")
            except ValueError:
                print("Nhập không hợp lệ.")
        
        # Tạo danh sách đối tượng đã phát hiện dạng đơn giản
        detected_objects = []
        for info in detection_info:
            obj_info = {
                'label': info['label'],
                'angle_2d': info['angle_2d']
            }
            
            # Thêm thông tin 3D nếu có
            if info['position_3d'] is not None:
                x_3d, y_3d, z_3d = info['position_3d']
                obj_info['position_3d'] = {
                    'x': x_3d,
                    'y': y_3d,
                    'z': z_3d
                }
            
            if info['angles_3d'] is not None:
                obj_info['angles_3d'] = info['angles_3d']
            
            detected_objects.append(obj_info) 
        # In danh sách đối tượng đã phát hiện      
        print("\nDanh sách đối tượng đã phát hiện:")
        for i, obj in enumerate(detected_objects):
            # Sử dụng .get() để tránh KeyError nếu 'distance' hoặc 'angles_3d' không tồn tại
            distance_str = f"{obj.get('position_3d', {}).get('z', 'không xác định'):.2f}m" if obj.get('position_3d') else "không xác định"
            angle_str = f"{obj.get('angles_3d', {}).get('azimuth', 0):.2f}°" if obj.get('angles_3d') else "không xác định"
            print(f"{i+1}. {obj['label']}: {distance_str}, Góc trong không gian 3D: {angle_str}")
        
        # Kết hợp ảnh kết quả và bản đồ độ sâu
        # Thay đổi kích thước của colormap để khớp chính xác với result_image
        colormap_resized = cv2.resize(colormap, (result_image.shape[1], result_image.shape[0]))
        
        # Đảm bảo rằng hai ảnh có cùng kích thước chính xác
        h1, w1 = result_image.shape[:2]
        h2, w2 = colormap_resized.shape[:2]
        
        # Nếu có sự khác biệt, điều chỉnh lại kích thước
        if h1 != h2 or w1 != w2:
            colormap_resized = cv2.resize(colormap_resized, (w1, h1))
        
        combined_image = cv2.hconcat([result_image, colormap_resized])
        
        # Hiển thị kết quả
        if show_result:
            cv2.namedWindow("Object Detection with Depth and Angle", cv2.WINDOW_NORMAL)
            cv2.imshow("Object Detection with Depth and Angle", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Lưu ảnh kết quả nếu cần
        if save_path:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            cv2.imwrite(save_path, combined_image)
            print(f"Đã lưu ảnh kết quả tại: {save_path}")
            
            # # Lưu thông tin phát hiện thành tệp JSON
            # json_path = os.path.splitext(save_path)[0] + "_detections.json"
            # with open(json_path, 'w', encoding='utf-8') as f:
            #     json.dump({
            #         'detailed_info': detection_info,
            #         'objects_list': detected_objects
            #     }, f, ensure_ascii=False, indent=4)
            # print(f"Đã lưu thông tin phát hiện tại: {json_path}")
        
        return combined_image, detected_objects
    
    def advice(self, detected_objects):
        """
        Phân tích các vật cản và đưa ra lời khuyên về hướng di chuyển an toàn
        
        Args:
            detected_objects: Danh sách các đối tượng đã phát hiện
            
        Returns:
            advice_str: Chuỗi chứa thông tin về vật cản và lời khuyên di chuyển
        """
        # Lọc bỏ các đối tượng là người
        obstacles = [obj for obj in detected_objects if obj['label'].lower() != 'person']
        
        # Sắp xếp các vật cản theo khoảng cách tăng dần
        obstacles_with_distance = []
        for obj in obstacles:
            # Kiểm tra xem có thông tin vị trí 3D không
            if obj.get('position_3d') is not None and obj['position_3d'].get('z') is not None:
                # Thêm vào danh sách với khoảng cách và góc
                distance = obj['position_3d']['z']
                angle = obj.get('angle_2d', 0)  # Góc 2D so với trung tâm
                obstacles_with_distance.append({
                    'label': obj['label'],
                    'distance': distance,
                    'angle': angle
                })
        
        # Sắp xếp theo khoảng cách
        obstacles_with_distance.sort(key=lambda x: x['distance'])
        
        # Tạo thông báo về các vật cản
        obstacle_info = []
        for i, obs in enumerate(obstacles_with_distance):
            distance = obs['distance']
            angle = obs['angle']
            side = "thẳng" if abs(angle) < 15 else "trái" if angle < 0 else "phải"
            
            obstacle_info.append(
                f"Vật cản gần bạn thứ {i+1} cách bạn {distance:.2f}m và nằm ở góc {abs(angle):.1f}° phía {side}"
            )
        
        # Kiểm tra các điều kiện hạn chế theo yêu cầu 
        straight_blocked = False
        left_blocked = False
        right_blocked = False
        
        for obs in obstacles_with_distance:
            distance = obs['distance']
            angle = obs['angle']
            
            # Điều kiện 1: Khoảng cách < 1.5m và |góc| < 15° -> phía thẳng không đi được
            if distance < 1.5 and abs(angle) < 15:
                straight_blocked = True
            
            # Điều kiện 2: Khoảng cách < 0.5m và |góc| > 15° -> bên trái/phải không đi được
            if distance < 0.5 and abs(angle) > 15:
                if angle < 0:  # Góc âm -> bên trái
                    left_blocked = True
                else:  # Góc dương -> bên phải
                    right_blocked = True
        
        # Tạo lời khuyên di chuyển
        advice_parts = []
        if straight_blocked:
            advice_parts.append("Không thể đi thẳng được")
        if left_blocked:
            advice_parts.append("Không đi sang bên trái được")
        if right_blocked:
            advice_parts.append("Không đi sang bên phải được")
        
        if not (straight_blocked or left_blocked or right_blocked):
            if obstacles_with_distance:
                advice_parts.append("Đường đi an toàn, có vật cản nhưng khoảng cách không gần")
            else:
                advice_parts.append("Không phát hiện vật cản nào")
        
        # Kết hợp thông tin vật cản và lời khuyên
        advice_str = "\n".join(obstacle_info)
        if advice_parts:
            advice_str += "\n\nGợi ý di chuyển: " + ", ".join(advice_parts)
        
        return advice_str

# def main():
#     # Tạo parser để nhận tham số dòng lệnh
#     parser = argparse.ArgumentParser(description='Phát hiện đối tượng và ước lượng khoảng cách với Depth Anything V2.')
#     parser.add_argument('--image', type=str, required=True, help='Đường dẫn đến ảnh đầu vào')
#     parser.add_argument('--yolo', type=str, default='yolo11n.pt', help='Đường dẫn đến model YOLOv11')
#     parser.add_argument('--output', type=str, default=None, help='Đường dẫn để lưu ảnh kết quả')
#     parser.add_argument('--calibrate', action='store_true', help='Bật chế độ hiệu chỉnh')
#     parser.add_argument('--no-show', action='store_true', help='Không hiển thị kết quả')
#     args = parser.parse_args()
    
#     # Khởi tạo bộ phát hiện kết hợp YOLOv11 và Depth Anything V2
#     estimator = DepthYOLOEstimator(yolo_model=args.yolo)
    
#     # Xử lý ảnh
#     output_path = args.output
#     if not output_path and not args.no_show:
#         output_path = os.path.splitext(args.image)[0] + "_result.jpg"
    
#     _, detected_objects = estimator.process_image(
#         image_path=args.image,
#         save_path=output_path,
#         show_result=not args.no_show,
#         calibrate=args.calibrate
#     )
    
#     # Hiển thị tổng kết
#     print(f"\nTổng cộng phát hiện được {len(detected_objects)} đối tượng.")

# if __name__ == "__main__":
#     main()