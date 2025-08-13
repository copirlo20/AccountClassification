# Nghiên cứu Graph Neural Networks (GNN) trong phân loại tài khoản mạng xã hội

## Mục tiêu
Nghiên cứu và đánh giá các giải pháp mạng nơ-ron đồ thị (Graph Neural Networks - GNN) nhằm phân loại tài khoản mạng xã hội dựa trên các đặc trưng đồ thị và hành vi người dùng.

## Nội dung chính
- **Tổng quan về GNN:** Khái niệm, các kiến trúc phổ biến (GCN, GraphSAGE, GAT, ...), ưu nhược điểm.
- **Phân loại tài khoản mạng xã hội:** Xây dựng mô hình GNN để dự đoán loại tài khoản (ví dụ: thật, giả, spam).
- **Tiền xử lý dữ liệu:** Biểu diễn mạng xã hội dưới dạng đồ thị, xác định nút và cạnh, trích xuất đặc trưng.
- **Huấn luyện và đánh giá mô hình:** Chọn thuật toán học, đánh giá độ chính xác, F1-score, AUC,...
- **So sánh với phương pháp truyền thống:** Logistic Regression, Random Forest, SVM,... trên cùng dữ liệu.

## Công cụ và thư viện
- Python
- PyTorch / PyTorch Geometric
- NetworkX, Pandas, NumPy, Scikit-learn

## Cách chạy
1. Chuẩn bị dữ liệu mạng xã hội dưới dạng đồ thị.
2. Cài đặt các thư viện cần thiết:  
   ```bash
    pip install torch torch-geometric networkx pandas scikit-learn
