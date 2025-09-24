# BÁO CÁO XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI LAND COVER

## 1. TỔNG QUAN DỰ LIỆU

### Đặc điểm dữ liệu:

- **Kích thước**: 900 mẫu với 10 thuộc tính
- **Số lớp**: 9 lớp phân loại (aquaculture, barren, croplands, forest, grassland, residential land, rice paddies, scrub, water)
- **Phân bố**: Cân bằng hoàn hảo (100 mẫu/lớp)
- **Features**: 7 kênh phổ từ ảnh vệ tinh Đà Nẵng (b1_da_nang đến b7_da_nang)
- **Tọa độ**: X, Y (được loại bỏ khỏi quá trình training)

### Tiền xử lý dữ liệu:

- Mã hóa nhãn lớp bằng LabelEncoder
- Chia dữ liệu: 80% training (720 mẫu) - 20% testing (180 mẫu)
- Stratified split để đảm bảo tỷ lệ các lớp đồng đều

## 2. MÔ HÌNH RANDOM FOREST

### Kiến trúc mô hình:

```python
RandomForestClassifier(
    n_estimators=100,      # 100 cây quyết định
    max_depth=None,        # Không giới hạn độ sâu
    min_samples_split=2,   # Tối thiểu 2 mẫu để phân chia
    min_samples_leaf=1,    # Tối thiểu 1 mẫu tại lá
    max_features='sqrt',   # Số features = √7 ≈ 2.6
    bootstrap=True,        # Sử dụng bootstrap sampling
    random_state=42,       # Đảm bảo tái tạo kết quả
    n_jobs=-1             # Sử dụng tất cả CPU cores
)
```

### Nguyên lý hoạt động:

- **Ensemble Learning**: Kết hợp 100 cây quyết định độc lập
- **Bootstrap Aggregating**: Mỗi cây được training trên subset ngẫu nhiên
- **Feature Randomness**: Mỗi node chỉ xem xét √7 features ngẫu nhiên
- **Voting**: Kết quả cuối cùng là majority vote của tất cả cây

## 3. MÔ HÌNH XGBOOST

### Kiến trúc mô hình:

```python
XGBClassifier(
    n_estimators=100,           # 100 boosting rounds
    max_depth=6,                # Độ sâu tối đa của cây
    learning_rate=0.1,          # Tốc độ học (eta)
    subsample=0.8,              # 80% mẫu cho mỗi cây
    colsample_bytree=0.8,       # 80% features cho mỗi cây
    random_state=42,            # Đảm bảo tái tạo kết quả
    eval_metric='mlogloss',     # Metric cho multiclass
    objective='multi:softprob'  # Multiclass với xác suất
)
```

### Nguyên lý hoạt động:

- **Gradient Boosting**: Xây dựng cây tuần tự, mỗi cây sửa lỗi của cây trước
- **Regularization**: Kiểm soát overfitting qua subsample và colsample
- **Optimization**: Sử dụng gradient descent để tối ưu loss function
- **Ensemble**: Kết hợp additive của tất cả weak learners

## 4. METRICS ĐÁNH GIÁ

### 4.1 Accuracy (Độ chính xác)

- **Công thức**: (True Predictions) / (Total Predictions)
- **Random Forest**: 67.22%
- **XGBoost**: 67.78%
- **Ý nghĩa**: Tỷ lệ dự đoán đúng trên tổng số dự đoán

### 4.2 Precision (Độ chính xác theo lớp)

- **Công thức**: True Positive / (True Positive + False Positive)
- **Random Forest**: 68.76% (weighted), 68.76% (macro)
- **XGBoost**: 68.40% (weighted), 68.40% (macro)
- **Ý nghĩa**: Trong số các mẫu được dự đoán thuộc lớp X, bao nhiêu % thực sự thuộc lớp X

### 4.3 Recall (Độ nhạy)

- **Công thức**: True Positive / (True Positive + False Negative)
- **Random Forest**: 67.22% (weighted), 67.22% (macro)
- **XGBoost**: 67.78% (weighted), 67.78% (macro)
- **Ý nghĩa**: Trong số các mẫu thực sự thuộc lớp X, bao nhiêu % được dự đoán đúng

### 4.4 F1-Score (Điểm F1)

- **Công thức**: 2 × (Precision × Recall) / (Precision + Recall)
- **Random Forest**: 67.24% (weighted), 67.24% (macro)
- **XGBoost**: 67.69% (weighted), 67.69% (macro)
- **Ý nghĩa**: Trung bình điều hòa của Precision và Recall

### 4.5 Confusion Matrix

Hiển thị chi tiết số lượng dự đoán đúng/sai cho từng lớp:

- **Đường chéo chính**: Dự đoán đúng
- **Các ô khác**: Nhầm lẫn giữa các lớp

## 5. PHÂN TÍCH FEATURE IMPORTANCE

### Random Forest:

1. **b5_da_nang**: 19.03% - Kênh cận hồng ngoại gần (Near-IR)
2. **b6_da_nang**: 18.61% - Kênh hồng ngoại sóng ngắn 1 (SWIR1)
3. **b7_da_nang**: 16.19% - Kênh hồng ngoại sóng ngắn 2 (SWIR2)
4. **b4_da_nang**: 14.03% - Kênh đỏ (Red)
5. **b3_da_nang**: 12.34% - Kênh xanh lá (Green)

### XGBoost:

1. **b5_da_nang**: 20.63% - Kênh cận hồng ngoại gần
2. **b4_da_nang**: 18.71% - Kênh đỏ
3. **b6_da_nang**: 18.62% - Kênh hồng ngoại sóng ngắn 1
4. **b7_da_nang**: 12.07% - Kênh hồng ngoại sóng ngắn 2
5. **b3_da_nang**: 11.68% - Kênh xanh lá

### Nhận xét:

- Cả hai mô hình đều xác định **b5_da_nang** (Near-IR) là feature quan trọng nhất
- Các kênh hồng ngoại (b5, b6, b7) có tầm quan trọng cao cho phân loại land cover
- Kênh b1_da_nang (Blue) có tầm quan trọng thấp nhất (~7-8%)

## 6. SO SÁNH KẾT QUẢ

| Metric             | Random Forest | XGBoost | Chênh lệch |
| ------------------ | ------------- | ------- | ---------- |
| Accuracy           | 67.22%        | 67.78%  | +0.56%     |
| Weighted Precision | 68.76%        | 68.40%  | -0.36%     |
| Weighted Recall    | 67.22%        | 67.78%  | +0.56%     |
| Weighted F1-Score  | 67.24%        | 67.69%  | +0.45%     |
| Macro F1-Score     | 67.24%        | 67.69%  | +0.45%     |

## 7. PHÂN TÍCH HIỆU SUẤT THEO LỚP

### Lớp có hiệu suất cao nhất:

- **Forest**: Cả hai mô hình đều đạt F1-Score > 85%
- **Water**: XGBoost đạt 75.68%, Random Forest đạt 74.29%
- **Barren**: Random Forest đạt 76.19%, XGBoost đạt 75.00%

### Lớp có hiệu suất thấp nhất:

- **Croplands**: Random Forest 48.78%, XGBoost 57.14%
- **Scrub**: Random Forest 61.90%, XGBoost 53.66%

### Nguyên nhân:

- Sự tương đồng về phổ giữa các lớp croplands, grassland, scrub
- Seasonal variations trong dữ liệu ảnh vệ tinh
- Complexity của urban land cover types

## 8. KẾT LUẬN VÀ KHUYẾN NGHỊ

### Kết luận:

1. **XGBoost** có hiệu suất tổng thể tốt hơn Random Forest (67.78% vs 67.22%)
2. Cả hai mô hình đều ổn định và có thể tái tạo kết quả
3. Feature importance nhất quán giữa hai mô hình
4. Hiệu suất phù hợp cho bài toán phân loại land cover với 9 lớp

### Khuyến nghị cải thiện:

1. **Feature Engineering**:

   - Tính toán các chỉ số thực vật (NDVI, SAVI, EVI)
   - Texture features từ Gray-Level Co-occurrence Matrix
   - Temporal features nếu có time series data

2. **Model Optimization**:

   - Hyperparameter tuning với GridSearch hoặc RandomSearch
   - Cross-validation để đánh giá robust hơn
   - Ensemble methods kết hợp cả Random Forest và XGBoost

3. **Data Augmentation**:

   - Thêm dữ liệu từ các khu vực khác
   - Synthetic data generation
   - Multi-temporal analysis

4. **Advanced Models**:
   - Deep Learning approaches (CNN, ResNet)
   - Transformer-based models cho remote sensing
   - Semi-supervised learning nếu có unlabeled data

### Ứng dụng thực tế:

- Giám sát thay đổi sử dụng đất
- Quy hoạch đô thị và nông nghiệp
- Đánh giá tác động môi trường
- Ứng phó biến đổi khí hậu
