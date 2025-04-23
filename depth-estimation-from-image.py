import cv2
import numpy as np
import torch
import argparse
import os
import json
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
    
    def process_image(self, image_path, save_path=None, show_result=True, calibrate=False):
        """
        Xử lý ảnh để phát hiện đối tượng và ước lượng khoảng cách
        
        Args:
            image_path: Đường dẫn đến ảnh đầu vào
            save_path: Đường dẫn để lưu ảnh kết quả (tùy chọn)
            show_result: Hiển thị kết quả
            calibrate: Bật chế độ hiệu chỉnh
            
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
        
        # Tạo một bản sao của ảnh để vẽ lên
        result_image = image.copy()
        
        # Phát hiện đối tượng với YOLOv11
        print("Đang phát hiện đối tượng...")
        detections = self.detect_objects(image)
        
        # Tính toán bản đồ độ sâu
        print("Đang tính toán bản đồ độ sâu...")
        depth_map, colormap = self.estimate_depth(image)
        
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
            
            # Đo khoảng cách đến đối tượng
            distance = self.measure_distance_to_bbox(depth_map, bbox)
            
            # Vẽ bbox và thông tin
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Tạo text hiển thị
            if distance:
                distance_text = f"{distance:.2f}m"
                text = f"{label} ({conf:.2f}): {distance_text}"
            else:
                text = f"{label} ({conf:.2f})"
            
            # Vẽ nhãn và khoảng cách
            cv2.rectangle(result_image, (x1, y1 - 30), (x1 + len(text) * 10, y1), color, -1)
            cv2.putText(result_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Lưu thông tin phát hiện
            detection_info.append({
                'label': label,
                'confidence': conf,
                'bbox': bbox.tolist(),
                'distance': distance if distance else "unknown"
            })
        
        # Hiệu chỉnh hệ số tỷ lệ nếu cần
        if calibrate and boxes.shape[0] > 0:
            print("\nChế độ hiệu chỉnh:")
            print("Các đối tượng được phát hiện:")
            for i, info in enumerate(detection_info):
                print(f"{i+1}. {info['label']} - Khoảng cách ước lượng: {info['distance']} m")
            
            try:
                choice = int(input("Chọn đối tượng để hiệu chỉnh (nhập số): ")) - 1
                if 0 <= choice < len(detection_info):
                    known_distance = float(input("Nhập khoảng cách thực đến đối tượng (mét): "))
                    new_scale = self.calibrate_scale_factor(depth_map, detection_info[choice]['bbox'], known_distance)
                    self.scale_factor = new_scale
                    print(f"Đã hiệu chỉnh hệ số tỷ lệ: {new_scale}")
                    
                    # Cập nhật lại khoảng cách cho tất cả đối tượng
                    for i, info in enumerate(detection_info):
                        distance = self.measure_distance_to_bbox(depth_map, info['bbox'])
                        detection_info[i]['distance'] = distance if distance else "unknown"
                    
                    # Vẽ lại ảnh kết quả với các khoảng cách đã cập nhật
                    result_image = image.copy()
                    for i, info in enumerate(detection_info):
                        bbox = info['bbox']
                        label = info['label']
                        conf = info['confidence']
                        distance = info['distance']
                        
                        # Vẽ bbox và thông tin
                        x1, y1, x2, y2 = map(int, bbox)
                        color = self.colors[i % len(self.colors)].tolist()
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Tạo text hiển thị
                        if distance != "unknown":
                            distance_text = f"{distance:.2f}m"
                            text = f"{label} ({conf:.2f}): {distance_text}"
                        else:
                            text = f"{label} ({conf:.2f})"
                        
                        # Vẽ nhãn và khoảng cách
                        cv2.rectangle(result_image, (x1, y1 - 30), (x1 + len(text) * 10, y1), color, -1)
                        cv2.putText(result_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    print("Lựa chọn không hợp lệ.")
            except ValueError:
                print("Nhập không hợp lệ.")
        
        # Tạo danh sách đối tượng đã phát hiện dạng đơn giản
        detected_objects = []
        for info in detection_info:
            detected_objects.append({
                'label': info['label'],
                'distance': info['distance'] if info['distance'] != "unknown" else None
            })
        
        # In ra danh sách đối tượng
        print("\nDanh sách đối tượng đã phát hiện:")
        for i, obj in enumerate(detected_objects):
            distance_str = f"{obj['distance']:.2f}m" if obj['distance'] is not None else "không xác định"
            print(f"{i+1}. {obj['label']}: {distance_str}")
        
        # Kết hợp ảnh kết quả và bản đồ độ sâu
        # Thay đổi kích thước của colormap để khớp với result_image
        colormap_resized = cv2.resize(colormap, (result_image.shape[1], result_image.shape[0]))
        combined_image = cv2.hconcat([result_image, colormap_resized])
        
        # Hiển thị kết quả
        if show_result:
            cv2.namedWindow("Object Detection with Depth", cv2.WINDOW_NORMAL)
            cv2.imshow("Object Detection with Depth", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Lưu ảnh kết quả nếu cần
        if save_path:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            cv2.imwrite(save_path, combined_image)
            print(f"Đã lưu ảnh kết quả tại: {save_path}")
            
            # Lưu thông tin phát hiện thành tệp JSON
            json_path = os.path.splitext(save_path)[0] + "_detections.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'detailed_info': detection_info,
                    'objects_list': detected_objects
                }, f, ensure_ascii=False, indent=4)
            print(f"Đã lưu thông tin phát hiện tại: {json_path}")
        
        return combined_image, detected_objects

def main():
    # Tạo parser để nhận tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Phát hiện đối tượng và ước lượng khoảng cách với Depth Anything V2.')
    parser.add_argument('--image', type=str, required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--yolo', type=str, default='yolo11n.pt', help='Đường dẫn đến model YOLOv11')
    parser.add_argument('--output', type=str, default=None, help='Đường dẫn để lưu ảnh kết quả')
    parser.add_argument('--calibrate', action='store_true', help='Bật chế độ hiệu chỉnh')
    parser.add_argument('--no-show', action='store_true', help='Không hiển thị kết quả')
    args = parser.parse_args()
    
    # Khởi tạo bộ phát hiện kết hợp YOLOv11 và Depth Anything V2
    estimator = DepthYOLOEstimator(yolo_model=args.yolo)
    
    # Xử lý ảnh
    output_path = args.output
    if not output_path and not args.no_show:
        output_path = os.path.splitext(args.image)[0] + "_result.jpg"
    
    _, detected_objects = estimator.process_image(
        image_path=args.image,
        save_path=output_path,
        show_result=not args.no_show,
        calibrate=args.calibrate
    )
    
    # Hiển thị tổng kết
    print(f"\nTổng cộng phát hiện được {len(detected_objects)} đối tượng.")

if __name__ == "__main__":
    main()