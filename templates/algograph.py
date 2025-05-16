# from graphviz import Digraph

# dot = Digraph(comment='Beam Search Algorithm')

# dot.node('A', 'Bắt đầu')
# dot.node('B', 'Khởi tạo:\nsequences = [[[], 0.0]]')
# dot.node('C', 'Lặp tối đa 42 bước')
# dot.node('D', 'Khởi tạo all_candidates = []')
# dot.node('E', 'Lặp từng (seq, score) trong sequences')
# dot.node('F', "seq kết thúc bằng 'endseq'?")
# dot.node('G1', "Thêm vào all_candidates")
# dot.node('G2', "Chuyển seq -> index + padding\nDự đoán từ tiếp theo\nLấy top beam_width từ\nTạo candidate mới\nTính score mới\nThêm vào all_candidates")
# dot.node('H', 'Sắp xếp all_candidates theo score')
# dot.node('I', 'Lấy top beam_width → sequences')
# dot.node('J', 'Tất cả seq kết thúc bằng endseq?')
# dot.node('K', 'Thoát vòng lặp')
# dot.node('L', 'Lấy sequence tốt nhất\nXóa startseq/endseq\nGhép thành câu\nThay từ đồng nghĩa')
# dot.node('M', 'Trả kết quả')
# dot.node('N', 'Kết thúc')

# dot.edges(['AB', 'BC', 'CD', 'DE'])
# dot.edge('E', 'F')
# dot.edge('F', 'G1', label='Có')
# dot.edge('F', 'G2', label='Không')
# dot.edge('G1', 'H')
# dot.edge('G2', 'H')
# dot.edge('H', 'I')
# dot.edge('I', 'J')
# dot.edge('J', 'K', label='Đúng')
# dot.edge('J', 'C', label='Sai')
# dot.edge('K', 'L')
# dot.edge('L', 'M')
# dot.edge('M', 'N')

# dot.render('beam_search_flowchart', format='png', view=True)
from graphviz import Digraph

def draw_testing_beam_search_flowchart():
    # Tạo đồ thị
    dot = Digraph(comment='TestingBeamSearch Flowchart', format='png')
    
    # Các bước chính trong thuật toán
    dot.node('A', 'Bước 1: Khởi tạo\nsequences = [[[], 0.0]]', shape='box')
    dot.node('B', 'Bước 2: Lặp qua từng vị trí trong chuỗi\n(for _ in range(42))', shape='diamond')
    dot.node('C', 'Bước 3: Xử lý từng chuỗi hiện tại\n(for seq, score in sequences)', shape='diamond')
    dot.node('D', 'Kiểm tra nếu seq[-1] == "endseq"', shape='diamond')
    dot.node('E', 'Thêm seq vào all_candidates', shape='box')
    dot.node('F', 'Chuyển đổi seq thành số (index)\nsequence = pad_sequences(...)', shape='box')
    dot.node('G', 'Dự đoán xác suất từ tiếp theo\nyhat = MyModel.predict(...)', shape='box')
    dot.node('H', 'Chọn beam_width từ có xác suất cao nhất\ntop_indices = np.argsort(yhat)[-beam_width:]', shape='box')
    dot.node('I', 'Tạo các ứng viên mới\n(candidate, candidate_score)', shape='box')
    dot.node('J', 'Thêm ứng viên vào all_candidates', shape='box')
    dot.node('K', 'Chọn beam_width chuỗi tốt nhất\nsequences = ordered[:beam_width]', shape='box')
    dot.node('L', 'Kiểm tra nếu tất cả seq[-1] == "endseq"', shape='diamond')
    dot.node('M', 'Kết thúc vòng lặp', shape='box')
    dot.node('N', 'Bước 7: Xử lý kết quả cuối cùng\n(final_sequence, result)', shape='box')
    dot.node('O', 'Bước 8: Xử lý từ đồng nghĩa\n(result.replace(...))', shape='box')
    dot.node('P', 'Trả về kết quả', shape='box')

    # Kết nối các bước
    dot.edges(['AB', 'BC', 'CD', 'DE', 'DF', 'FG', 'GH', 'HI', 'IJ', 'JK', 'KL', 'LM'])
    dot.edge('D', 'E', label='True')
    dot.edge('D', 'F', label='False')
    dot.edge('L', 'M', label='True')
    dot.edge('L', 'B', label='False')
    dot.edge('M', 'N')
    dot.edge('N', 'O')
    dot.edge('O', 'P')

    # Lưu và hiển thị lưu đồ
    dot.render('TestingBeamSearch_Flowchart', view=True)

# Gọi hàm để vẽ lưu đồ
draw_testing_beam_search_flowchart()