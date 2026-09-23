# Hệ CSDL lưu trữ và tìm kiếm ảnh cây thân gỗ (CBIR)

Bài tập lớn môn **Hệ cơ sở dữ liệu đa phương tiện**: xây dựng bộ dữ liệu ≥ 500 ảnh cây thân gỗ trưởng thành,
trích rút nhiều bộ đặc trưng (màu sắc, hình dạng, kết cấu/bố cục, embedding), quản trị đặc trưng trong CSDL
SQLite và tìm 5 ảnh giống nhất với một ảnh cây mới.

Báo cáo đầy đủ: `docs/BaoCao_HeCSDL_AnhCay.docx` / `docs/BaoCao_HeCSDL_AnhCay.pdf`.
Hướng dẫn dùng demo web: `docs/HUONG_DAN_DEMO.md`.

## Cấu trúc

```
data/images/          ảnh CSDL đã chuẩn hoá 512×512 (cây nằm giữa ảnh)      data/manifest.csv
data/queries/         10 ảnh truy vấn thử (không nằm trong CSDL)              data/queries.csv
db/tree_features.sqlite   CSDL đặc trưng (images, feature_sets, features, feature_stats, indexes, ...)
src/treedb/           mã nguồn: collect, preprocess, segment, features/{color,shape,texture,deep}, db, index, search, evaluate, visualize
scripts/              build_dataset → extract_features → build_index → run_evaluation → make_figures → build_docs; query.py
app/streamlit_app.py  demo giao diện web
results/              evaluation.json, index_meta.json, query_examples/
docs/figures/         hình minh hoạ dùng trong báo cáo
```

## Cài đặt và chạy

```bash
pip install -r requirements.txt

python scripts/build_dataset.py        # 1. sưu tầm (Wikimedia Commons) + chuẩn hoá ảnh
python scripts/extract_features.py     # 2-3. phân đoạn, trích rút đặc trưng, ghi CSDL SQLite
python scripts/build_index.py          # 3b. chỉ mục kd-tree (PCA) + phân cụm K-means
python scripts/run_evaluation.py       # 4c. đánh giá leave-one-out (P@5, mAP, so sánh backend)
python scripts/make_figures.py         # hình minh hoạ + ví dụ truy vấn
python scripts/build_docs.py           # sinh báo cáo docx + pdf

python scripts/query.py --image data/queries/query_01.jpg --backend linear --fig out.png   # truy vấn 1 ảnh
streamlit run app/streamlit_app.py                                                            # demo web
```

Bộ dữ liệu, CSDL và kết quả đánh giá đã được sinh sẵn trong repo, có thể chạy thẳng bước truy vấn / demo.
