# Graph Neural Networks (GNN) trong phân loại tài khoản mạng xã hội

## Mục tiêu
Nghiên cứu và đánh giá các giải pháp mạng nơ-ron đồ thị (Graph Neural Networks - GNN) nhằm phân loại tài khoản mạng xã hội dựa trên các đặc trưng đồ thị và hành vi người dùng.

## Nội dung chính
- **Tổng quan về GNN:** Khái niệm, các kiến trúc phổ biến (GCN, GraphSAGE, GAT, ...), ưu nhược điểm.
- **Phân loại tài khoản mạng xã hội:** Xây dựng các mô hình GNN để dự đoán loại tài khoản (Real - Fake).
- **Tiền xử lý dữ liệu:** Biểu diễn mạng xã hội dưới dạng đồ thị, xác định nút và cạnh, trích xuất đặc trưng.
- **Huấn luyện và đánh giá mô hình:** Đánh giá qua các chỉ số Accuracy, Precision, Recall, F1-score, AUC-ROC,...
- **So sánh với phương pháp truyền thống:** XGBoost, AdaBoost, Random Forest, Decision Tree, SVM,...

## Công cụ và thư viện
- Python
- PyTorch / PyTorch Geometric
- NetworkX, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- gender-guesser (Nếu cần dự đoán giới tính)
- ipykernel (Nếu cần xuất HTML từ Jupyter Notebook)
- FastAPI (Cho ứng dụng BackEnd)
- NodeJs (Cho ứng dụng FrontEnd)

## Cách chạy
1. Xem các nghiên cứu:  
    - Các mô hình GNN trong folder `GNN` (các file `.ipynb`).  
    - Các phương pháp Machine Learning khác trong folder `MachineLearning` (các file `.ipynb`).
2. Cài đặt các thư viện cần thiết:  
    ```bash
    pip install torch torch-geometric networkx pandas scikit-learn matplotlib seaborn fastapi uvicorn gender-guesser ipykernel
    ```
3. Chạy BackEnd:  
    ```bash
    uvicorn main:API --reload
    ```
4. Chạy FrontEnd:  
    ```bash
    cd app
    # Nếu đây là lần chạy đầu tiên hoặc folder node_modules chưa tồn tại, chạy:
    npm install
    # Sau đó chạy ứng dụng:
    npm start
    ```
