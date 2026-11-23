"""Bao cao 2 - muc 2: co so ly thuyet ap dung tu bai giang (TS. Nguyen Dinh Hoa).

Trich dan dang [1, bai X, tr. Y]: Y la so trang in tren slide (khong phai so trang cua tep PDF).
Cac ky thuat KHONG co trong slide (PCA, k-means++, P@k/mAP, cong thuc ket hop co trong so, SegFormer,
MobileNetV2...) duoc ghi ro la phan mo rong cua nhom kem tai lieu ngoai.
"""
from __future__ import annotations

from docx_helpers import Report

REFERENCES = [
    "Nguyễn Đình Hóa, Bài giảng Hệ cơ sở dữ liệu đa phương tiện, Học viện Công nghệ Bưu chính Viễn thông: "
    "bài 1 (Giới thiệu), bài 2 (Các loại dữ liệu ĐPT), bài 4 (Kiến trúc hệ CSDL ĐPT), bài 5 (Mô hình hóa dữ liệu "
    "ĐPT), bài 6 (Siêu dữ liệu và CSDL ĐPT), bài 7 (Cấu trúc dữ liệu đa chiều), bài 8 (Truy vấn không gian vector "
    "và phân cụm), bài 11 (Chỉ số hóa và truy vấn dữ liệu ảnh). Số trang trích dẫn là số in trên slide.",
    "J. L. Bentley, “Multidimensional binary search trees used for associative searching,” Communications of the "
    "ACM, vol. 18, no. 9, 1975.",
    "A. Guttman, “R-trees: A dynamic index structure for spatial searching,” Proc. ACM SIGMOD, 1984.",
    "D. Arthur, S. Vassilvitskii, “k-means++: The advantages of careful seeding,” Proc. ACM-SIAM SODA, 2007.",
    "R. M. Haralick, K. Shanmugam, I. Dinstein, “Textural features for image classification,” IEEE Trans. "
    "Systems, Man, and Cybernetics, 1973.",
    "T. Ojala, M. Pietikäinen, T. Mäenpää, “Multiresolution gray-scale and rotation invariant texture "
    "classification with local binary patterns,” IEEE Trans. PAMI, vol. 24, no. 7, 2002.",
    "H. Tamura, S. Mori, T. Yamawaki, “Textural features corresponding to visual perception,” IEEE Trans. Systems, "
    "Man, and Cybernetics, 1978.",
    "M.-K. Hu, “Visual pattern recognition by moment invariants,” IRE Trans. Information Theory, 1962.",
    "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, “MobileNetV2: Inverted residuals and linear "
    "bottlenecks,” Proc. CVPR, 2018.",
    "E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo, “SegFormer: Simple and efficient design for "
    "semantic segmentation with transformers,” Proc. NeurIPS, 2021.",
    "C. D. Manning, P. Raghavan, H. Schütze, Introduction to Information Retrieval, Cambridge University Press, "
    "2008 (chương 8: P@k, MAP; chương 9: phản hồi liên quan Rocchio).",
]


def _mmdb(r: Report):
    r.h(2, "2.1. Hệ CSDL đa phương tiện và truy vấn ảnh dựa trên nội dung")
    r.p("Bài giảng phân biệt CSDL truyền thống, vốn “chủ yếu trả lời các truy vấn khớp chính xác về thuộc tính có "
        "cấu trúc”, với CSDL đa phương tiện, nơi “các truy vấn thường là tìm kiếm dữ liệu tương đồng” và “kết quả "
        "thường được xếp hạng thay vì chỉ lọc các bản ghi” [1, bài 1, tr. 14]. Tìm kiếm chính xác hợp với siêu dữ "
        "liệu (tác giả, định dạng…), tìm kiếm tương đồng “so sánh các vector đặc trưng và trả về các kết quả gần "
        "giống nhất”, và “hầu hết các hệ thống MMDB thực tiễn kết hợp cả hai dạng tìm kiếm trong một quy trình truy "
        "vấn duy nhất” [1, bài 1, tr. 20]. Hệ của nhóm đi theo đúng hướng này: lọc theo siêu dữ liệu bằng SQL và "
        "xếp hạng theo khoảng cách vector.")
    r.p("Trong bốn kỹ thuật chỉ số hoá ảnh, “kỹ thuật thứ 2 và thứ 4 được gọi là truy vấn dữ liệu ảnh dựa trên nội "
        "dung”, trong đó kỹ thuật thứ 4 “sử dụng các thuộc tính bậc thấp (mầu sắc, ngữ cảnh) để đánh chỉ số và truy "
        "vấn dữ liệu” [1, bài 11, tr. 3]. Thuộc tính bậc thấp gồm màu sắc, hình dạng và phân bố màu nền/độ tương "
        "phản [1, bài 2, tr. 15], tương ứng với ba bộ đặc trưng thủ công của nhóm. Giới hạn của chúng là “khoảng "
        "cách ngữ nghĩa”, tức “sự khác biệt giữa thông tin mô tả bởi máy móc và tư duy con người về cùng một đối "
        "tượng đa phương tiện” [1, bài 1, tr. 29]; bài giảng nêu rằng thuộc tính bậc cao “cần có mô hình, thẻ, hoặc "
        "các đặc trưng sâu” [1, bài 2, tr. 17], đây là lý do nhóm bổ sung bộ embedding sâu.")
    r.p("Theo các đặc tính ở [1, bài 1, tr. 40], hệ của nhóm là hệ tĩnh (CSDL nạp một lần, nhiều truy vấn) và thuộc "
        "loại tìm kiếm theo chức năng (tra cứu theo nội dung, không chỉ theo siêu dữ liệu).")


def _lifecycle(r: Report):
    r.h(2, "2.2. Vòng đời đối tượng đa phương tiện và mô hình truy vấn hai giai đoạn")
    r.p("Bài giảng mô tả vòng đời của một đối tượng đa phương tiện qua sáu bước [1, bài 1, tr. 15]. Báo cáo này "
        "trình bày bốn bước đầu ở mục 3 và hai bước cuối ở mục 4:")
    r.table(["Bước theo bài giảng", "Cài đặt trong hệ thống", "Mục"], [
        ["1. Thu thập đối tượng thô", "Ảnh từ Wikimedia Commons, kiểm duyệt bằng mắt", "Báo cáo 1"],
        ["2. Tiền xử lý và chuẩn hoá", "512×512, khử trùng lặp, tách nền", "3.1"],
        ["3. Trích xuất siêu dữ liệu và đặc trưng", "Siêu dữ liệu nguồn + 4 bộ đặc trưng", "3.2"],
        ["4. Lưu dữ liệu thô cùng mô tả dẫn xuất", "Tệp ảnh + CSDL SQLite + chỉ mục", "3.3 – 3.5"],
        ["5. Chạy truy vấn tìm kiếm chính xác hoặc tương tự", "Lọc siêu dữ liệu + tìm kiếm tương đồng", "4.1, 4.2"],
        ["6. Hiển thị kết quả theo xếp hạng, thu thập phản hồi", "Top-5 giảm dần theo sim, phản hồi Rocchio",
         "4.2 – 4.4"],
    ], widths=[5.6, 7.4, 3.0], caption="Vòng đời đối tượng đa phương tiện [1, bài 1, tr. 15] trong hệ thống",
        size=10.5)
    r.p("Mô hình truy vấn ảnh theo nội dung có hai giai đoạn [1, bài 1, tr. 24–25]: tạo CSDL (thu thập ảnh → số "
        "hoá → phân tích ảnh và trích xuất đặc trưng → CSDL ảnh) và truy vấn CSDL (ảnh truy vấn → số hoá → phân "
        "tích và trích xuất đặc trưng → tìm kiếm tương đồng → kết quả), trong đó ảnh truy vấn được “phân tích giống "
        "như thực hiện với các đối tượng trong cơ sở dữ liệu”. Sơ đồ chi tiết [1, bài 1, tr. 26–27] gồm năm khối: "
        "1. đưa vào CSDL, 2. trích xuất đặc trưng, 3. chuẩn bị truy vấn (chuẩn hoá, phân đoạn, trích xuất đặc "
        "trưng, tối ưu thành kế hoạch truy vấn), 4. tính độ tương đồng và xử lý truy vấn (dùng chỉ mục đặc trưng), "
        "5. chuẩn bị kết quả; mũi tên phản hồi liên quan đi từ kết quả trở lại khối chuẩn bị truy vấn. Mục 4.1 "
        "ánh xạ từng khối của nhóm vào năm khối này.")


def _metadata(r: Report):
    r.h(2, "2.3. Mô hình dữ liệu và siêu dữ liệu")
    r.p("Mô hình lưu trữ được bài giảng gọi là mô hình lai: lưu toàn vẹn tệp gốc, lưu siêu dữ liệu (tiêu đề, tác "
        "giả, định dạng…) và các thuộc tính trích xuất (biểu đồ màu, embedding…); “chỉ mục thường được xây dựng "
        "trên siêu dữ liệu cùng các đặc trưng, không trực tiếp trên các dữ liệu gốc”, và mô hình này “hỗ trợ cả "
        "truy vấn lọc bản ghi theo SQL tiêu chuẩn và truy vấn dựa trên nội dung”, đồng thời cho phép “cập nhật các "
        "đặc trưng pipeline mà không mất đối tượng đa phương tiện chính” [1, bài 1, tr. 18]. Dữ liệu gốc không nên "
        "nằm trong các hàng quan hệ mà được “tham chiếu bởi URI/OID” [1, bài 2, tr. 42].")
    r.p("Với mô hình hướng đối tượng, “một tập thực thể có thể coi là một miền không gian đa chiều trong đó mỗi thực "
        "thể là một tập giá trị cụ thể nào đó”, mô tả bằng tập thực thể Oid, tập lớp Cid và tập thuộc tính Att "
        "[1, bài 5, tr. 9–10]. Trong hệ của nhóm: Oid là image_id, Cid là chi cây và nhóm lá rộng/lá kim, Att là "
        "các thuộc tính ảnh; mỗi ảnh là một điểm trong không gian đặc trưng nhiều chiều. Nhóm dùng mô hình quan "
        "hệ (SQLite) và tránh nhược điểm “phẳng” của nó bằng cách tách bảng siêu dữ liệu và bảng vector.")
    r.p("Siêu dữ liệu hình ảnh có ba loại [1, bài 6, tr. 15]: văn bản mô tả nội dung, các thuộc tính lưu bằng mô hình "
        "quan hệ, và “siêu dữ liệu dựa trên nội dung: siêu dữ liệu bao gồm tất cả các đặc trưng ảnh”. Ba mức diễn "
        "tả của Panofsky [1, bài 6, tr. 26–30] cho thấy hệ thống phủ tới đâu:")
    r.table(["Mức diễn tả (Panofsky)", "Ý nghĩa theo bài giảng", "Trong hệ thống"], [
        ["Miêu tả", "Thông tin cơ bản, tự nhiên của thực thể, ai cũng nhận biết", "4 vector đặc trưng (màu, hình "
                                                                               "dạng, kết cấu, embedding)"],
        ["Nhận diện", "Tên của thực thể, phân biệt các thực thể với nhau", "Chi cây (genus), nhóm lá, tiêu đề nguồn"],
        ["Truyền đạt", "Ý nghĩa ẩn, gắn với văn hoá, do chuyên gia thực hiện", "Không có: hệ thống không tự suy ra "
                                                                             "được (giới hạn ngữ nghĩa)"],
    ], widths=[3.6, 6.4, 6.0], caption="Ba mức diễn tả của siêu dữ liệu và mức độ hệ thống đáp ứng", size=10.5)
    r.p("Về kiến trúc, hệ đi theo kiến trúc mở của CSDL đa phương tiện [1, bài 6, tr. 33]: bộ quản lý siêu dữ liệu "
        "(SQLite) và bộ quản lý tệp đa phương tiện (thư mục ảnh) tách riêng, nối với nhau bằng một module tích hợp "
        "(src/treedb/db.py và src/treedb/search.py), thay cho một MM-DBMS khép kín.")


def _vector(r: Report):
    r.h(2, "2.4. Không gian vector, biểu đồ màu và độ đo tương đồng")
    r.p("Trong mô hình không gian vector, mỗi đối tượng là một vector Dᵢ = [dᵢ₁, …, dᵢₙ] và câu truy vấn là "
        "Qⱼ = [qⱼ₁, …, qⱼₙ] [1, bài 8, tr. 3]; với ảnh, mỗi “từ khoá” là một chiều đặc trưng. Bài giảng đưa ra "
        "các độ đo [1, bài 8, tr. 7 và 18]:")
    r.bullets([
        "Lp(Dᵢ, Dⱼ) = (Σₖ |dᵢₖ − dⱼₖ|ᵖ)^(1/p); L1(Dᵢ, Dⱼ) = Σₖ |dᵢₖ − dⱼₖ|; L2(Dᵢ, Dⱼ) = √(Σₖ (dᵢₖ − dⱼₖ)²).",
        "Công thức chuẩn hoá S(Dᵢ, Qⱼ) = Σₖ dᵢₖqⱼₖ / (√Σₖ dᵢₖ² · √Σₖ qⱼₖ²), tức độ tương đồng cosine; “kết quả truy "
        "vấn được sắp xếp theo thứ tự giảm dần về sự tương đồng”. Nhóm dùng khoảng cách cosine = 1 − S.",
    ])
    r.p("Biểu đồ tần suất màu H(M) = [h₁, …, hₙ], hⱼ là số điểm ảnh có mức màu j, được lưu và truy vấn bằng kỹ thuật "
        "không gian vector hoặc phân cụm, với độ đo L1, L2 [1, bài 11, tr. 9–10]. Các kỹ thuật nâng cao mà nhóm cài "
        "đặt: biểu đồ luỹ tích chᵢ = Σ_{j ≤ i} hⱼ và biểu đồ theo trọng số PWH [1, bài 11, tr. 15], biểu đồ theo "
        "vùng để giữ quan hệ không gian [1, bài 11, tr. 16], đổi RGB sang không gian màu cảm nhận [1, bài 11, tr. 18]. "
        "Bài giảng cảnh báo “hiệu ứng che mặt”: “các ảnh có mầu nền giống nhau sẽ mang nội dung giống nhau, mặc dù "
        "các vật thể trong đó là khác hẳn nhau”, và đề xuất biểu đồ riêng cho vật thể và hình nền, truy vấn theo "
        "vật thể, theo nền hoặc kết hợp có trọng số [1, bài 11, tr. 17]. Nhóm tách hẳn nền và mô tả chính cái cây.")
    r.p("Hình dạng dùng trục chính, trục phụ, hình chữ nhật cơ bản, tâm sai [1, bài 11, tr. 20]; lưới ô vuông với ô "
        "có trên 15% diện tích vật thể nhận giá trị 1, đọc trái sang phải, trên xuống dưới [1, bài 11, tr. 21]; "
        "moment bất biến và hệ số Fourier [1, bài 11, tr. 25]. Bố cục/kết cấu gồm độ thô, độ tương phản, hướng sắp "
        "xếp các nét [1, bài 11, tr. 26]; đánh chỉ số trên miền nén dùng hệ số DCT [1, bài 11, tr. 27].")
    r.p("Kết hợp nhiều bộ đặc trưng: bài giảng chỉ có tiền lệ kết hợp có trọng số giữa biểu đồ vật thể và hình nền "
        "[1, bài 11, tr. 17]. Công thức kết hợp D = Σ_s w_s·d_s/scale_s ở mục 3.4 là do nhóm đề xuất theo tinh thần "
        "đó. Phản hồi liên quan theo bài giảng: Qⱼ₊₁ = Qⱼ + α·Σ_{D ∈ Rel} D − β·Σ_{D ∈ NonRel} D [1, bài 8, tr. 13]; "
        "nhóm dùng dạng Rocchio lấy trung bình và chia cho tổng hệ số [11] để vector mới giữ cùng thang đo.")


def _index(r: Report):
    r.h(2, "2.5. Cấu trúc dữ liệu đa chiều và phân cụm")
    r.p("Mục tiêu của cấu trúc dữ liệu đa chiều là “lưu trữ và tra cứu dữ liệu theo không gian đa chiều” "
        "[1, bài 7, tr. 3]. Các cấu trúc nhóm cài đặt:")
    r.bullets([
        ("Cây k-d: ", "cây phân nhóm dữ liệu theo không gian bằng phương pháp đệ quy, mỗi nút chia miền thành hai "
                      "nửa bằng mặt phẳng vuông góc với một trục, thứ tự trục thay đổi theo cấp [1, bài 7, tr. 9, "
                      "12]; biến thể chia theo giá trị trung bình của thuộc tính [1, bài 7, tr. 18]; tra cứu láng "
                      "giềng gần nhất NN và k-NN “chọn và giữ k điểm kết quả tốt nhất” [1, bài 7, tr. 19–21]."),
        ("Cây tứ phân vùng điểm PR: ", "mỗi nút chia miền thành 4 phần NW, NE, SW, SE, các miền được chia tới khi "
                                       "mỗi miền con còn một (hoặc b) điểm [1, bài 7, tr. 23, 42, 47]; nhược điểm: "
                                       "“biểu diễn trên miền dữ liệu đa chiều tốn nhiều dung lượng lưu trữ” "
                                       "[1, bài 7, tr. 48]."),
        ("Cây R: ", "lưu dữ liệu bằng vùng hình chữ nhật, mỗi nút là hình chữ nhật bao nhỏ nhất MBR của các con, "
                    "mỗi nút chứa từ K/2 đến K con [1, bài 7, tr. 50–51]; “hữu ích trong việc lưu trữ dữ liệu lớn, có "
                    "ít thuộc tính (thường tối đa 10 thuộc tính)” [1, bài 7, tr. 50]."),
        ("K-means: ", "khởi tạo K điểm gốc; gán mỗi bản ghi vào nhóm có điểm gốc gần nhất; đặt điểm gốc mới là "
                      "điểm trung bình của nhóm; lặp tới khi nhóm không đổi, điểm gốc không đổi hoặc vượt số vòng "
                      "lặp cho trước [1, bài 8, tr. 26]. Khi truy vấn, cụm gần nhất được xác định theo “khoảng cách "
                      "giữa câu truy vấn với điểm trung bình của cụm”; cụm ít dữ liệu thì lấy toàn bộ, cụm nhiều "
                      "dữ liệu thì chỉ lấy các bản ghi giống nhất [1, bài 8, tr. 28]."),
    ])
    r.p("Vector đặc trưng của nhóm có hàng trăm chiều, trong khi cây R và cây tứ phân chỉ hiệu quả ở số chiều thấp. "
        "Vì vậy nhóm giảm chiều bằng phân tích thành phần chính PCA (không có trong bài giảng) trước khi dựng cây: "
        "8 chiều cho cây R (đúng giới hạn “tối đa 10 thuộc tính”), 2 chiều cho cây tứ phân. Khởi tạo K-means dùng "
        "k-means++ [4] thay cho chọn ngẫu nhiên để kết quả ổn định hơn.")


def _evaluation(r: Report):
    r.h(2, "2.6. Đánh giá hệ thống truy vấn")
    r.p("Bài giảng tách hai mặt đánh giá [1, bài 1, tr. 44–45]: độ hiệu quả về tài nguyên (efficiency: tốc độ xử lý, "
        "thời gian phản hồi, bộ nhớ, số thao tác vào ra) và sự hiệu quả về kết quả (effectiveness: kết quả có đúng "
        "với yêu cầu truy vấn hay không); “một hệ thống phản hồi rất nhanh nhưng kết quả tìm kiếm không thực sự liên "
        "quan sẽ không đạt, hoặc ngược lại”. Với một truy vấn, CSDL chia thành bốn phần theo ma trận chất lượng "
        "[1, bài 1, tr. 47–48]: lựa chọn đúng ca, lựa chọn sai fa, loại bỏ sai fd, loại bỏ đúng cd; từ đó tỉ lệ "
        "đúng P = ca/(ca + fa) và độ phủ R = ca/(ca + fd) [1, bài 1, tr. 49]. “Cả tỉ lệ đúng và độ phủ chỉ có ý "
        "nghĩa khi chúng được sử dụng đồng thời” [1, bài 1, tr. 50]; đánh giá ngoại tuyến dùng “các truy vấn cố "
        "định và dựa vào nhãn kết quả liên quan” [1, bài 1, tr. 51].")
    r.p("Vì hệ luôn trả về đúng 5 ảnh, nhóm dùng thêm các độ đo xếp hạng phổ biến trong truy hồi thông tin [11]: "
        "P@5 (tỉ lệ đúng P tính trên 5 kết quả đầu), P@1 và mAP@5 (trung bình độ chính xác tại các vị trí có kết "
        "quả đúng, có tính đến thứ tự). Đây là phần mở rộng của nhóm dựa trên P, R của bài giảng.")


def sec_theory(r: Report):
    r.h(1, "2. Cơ sở lý thuyết áp dụng")
    r.p("Mục này tóm tắt các khái niệm của học phần được dùng trực tiếp trong thiết kế hệ thống, kèm trích dẫn bài "
        "giảng [1]. Các kỹ thuật nằm ngoài bài giảng được ghi rõ là phần mở rộng của nhóm và có tài liệu tham khảo "
        "riêng.")
    _mmdb(r)
    _lifecycle(r)
    _metadata(r)
    _vector(r)
    _index(r)
    _evaluation(r)
