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
                angle = obj.get('angles_3d', {}).get('azimuth', 0) # Góc 3D so với trung tâm
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