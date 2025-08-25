# Graph Neural Networks (GNN) trong phân loại tài khoản mạng xã hội

## Mục tiêu
Nghiên cứu và đánh giá các giải pháp mạng nơ-ron đồ thị (Graph Neural Networks - GNN) nhằm phân loại tài khoản mạng xã hội dựa trên các đặc trưng đồ thị và hành vi người dùng.

## Nội dung chính
- **Tổng quan về GNN:** Khái niệm, các kiến trúc phổ biến (GCN, GraphSAGE, GAT, ...), ưu nhược điểm.
- **Phân loại tài khoản mạng xã hội:** Xây dựng các mô hình GNN để dự đoán loại tài khoản (Real - Fake).
- **Tiền xử lý dữ liệu:** Biểu diễn mạng xã hội dưới dạng đồ thị, xác định nút và cạnh, trích xuất đặc trưng.
- **Huấn luyện và đánh giá mô hình:** Đánh giá qua các chỉ số Accuracy, Precision, Recall, F1-score, AUC-ROC,...

## Công cụ và thư viện
- Python
- PyTorch / PyTorch Geometric
- NetworkX, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- ipykernel (Nếu cần xuất HTML từ Jupyter Notebook)
- FastAPI (Cho ứng dụng BackEnd)
- NodeJs (Cho ứng dụng FrontEnd)

## Cách chạy
1. Quá trình huấn luyện các mô hình GNN được lưu trong folder `GNN` (các file `.ipynb`).
2. Các mô hình sau huấn luyện được lưu trong folder `model` (các file `.pth`)
3. Cài đặt các thư viện cần thiết:  
    ```bash
    pip install torch torch-geometric networkx scikit-learn matplotlib seaborn fastapi uvicorn ipykernel
    ```
4. Chạy BackEnd:  
    ```bash
    uvicorn main:API --reload
    ```
5. Chạy FrontEnd:  
    ```bash
    cd app
    # Cài đặt node_modules:
    npm install
    # Chạy ứng dụng:
    npm start
    ```
