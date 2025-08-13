import pandas as pd
import matplotlib.pyplot as plt
from graph import Graph
from model import USER_PATHS, USEFUL_COLS, encode_users, GAT, GCN, GraphSAGE, Predictions
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Đọc dữ liệu người dùng và lấy các cột cần thiết
users_raw = pd.concat([pd.read_csv(path) for path in USER_PATHS], ignore_index=True)
labels = users_raw['dataset'].apply(lambda x: 0 if x in ['TFP', 'E13', 0] else 1)
users_raw = users_raw[USEFUL_COLS]

# Hàm tính độ chính xác của dự đoán
def accuracy(predict):
    return (predict == labels.to_numpy()).sum() / labels.shape[0]

# Hàm vẽ biểu đồ confusion matrix
def chart(predictions, name, color, index):
    results = predictions.predict().argmax(dim=1).numpy()
    cm = confusion_matrix(labels.to_numpy(), results)
    class_names = ['Real', 'Fake']
    plt.subplot(1, 3, index)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.ylabel('True Label')
    plt.title(f'{name} {accuracy(results)*100:.2f}%\n', fontsize=11)

# Tải dữ liệu người dùng và xây dựng đồ thị
users = encode_users(users_raw)
graph = Graph(users).build()

# Sử dụng các mô hình để dự đoán
gcnModel = GCN(graph.num_node_features, 64, 2)
gcn = Predictions(gcnModel, './model/GCN.pth', graph)

gatModel = GAT(graph.num_node_features, 64, 2)
gat = Predictions(gatModel, './model/GAT.pth', graph)

graphSAGEModel = GraphSAGE(graph.num_node_features, 64, 2)
graphSAGE = Predictions(graphSAGEModel, './model/GraphSAGE.pth', graph)

# Biểu đồ kết quả dự đoán
fig = plt.figure(figsize=(12, 3))
manager = plt.get_current_fig_manager()
manager.window.state('zoomed')
chart(gcn, 'GCN', 'Blues', 1)
chart(gat, 'GAT', 'Reds', 2)
chart(graphSAGE, 'GraphSAGE', 'Greens', 3)
plt.suptitle('Model Evaluation on CRESCI-2015 Dataset', fontsize=16)
plt.tight_layout()
plt.show()