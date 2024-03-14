# NhanDienVatThe_Object-Detection_MachineLearning
Nhận Diện Vật Thể (Object Detection)
PHẦN NỘI DUNG
I.	Tổng Quan Các Thuật Toán: 
Ứng dụng các thuật toán học tăng cường trong việc nhận diện ảnh và video đã mang lại nhiều tiềm năng trong thời gian gần đây. Bằng cách sử dụng các thuật toán học tăng cường, ta có thể cải thiện độ chính xác của các mô hình nhận diện ảnh và video, đồng thời tăng tính linh hoạt và độ bền của chúng.
Các thuật toán học tăng cường có thể được sử dụng để tăng độ phân giải của ảnh, điều chỉnh độ sáng và độ tương phản của ảnh, loại bỏ nhiễu, hoặc tăng độ chi tiết của ảnh. Bằng cách tăng độ phân giải của ảnh, ta có thể giảm độ mờ của đối tượng trong ảnh, nâng cao độ chính xác và độ tin cậy của mô hình nhận diện.
Các thuật toán học tăng cường cũng có thể được sử dụng để cải thiện độ chính xác của các mô hình nhận diện video. Việc sử dụng các thuật toán này có thể giúp giảm độ nhầy của video, loại bỏ nhiễu và tăng độ nét của video. Các kỹ thuật này có thể giúp tăng khả năng phát hiện đối tượng, theo dõi đối tượng và phân tích hành vi đối tượng trong video.
Một số ví dụ về các thuật toán học tăng cường được sử dụng trong việc nhận diện ảnh và video bao gồm:
1.	Linear Regression: 
Là một thuật toán học có giám sát trong machine learning, được sử dụng để dự đoán giá trị của một biến số dựa trên các biến số độc lập. Trong nhận diện ảnh và video, Linear Regression có thể được sử dụng để giảm nhiễu, cải thiện độ tương phản và độ sáng của ảnh hoặc video. Trong công nghệ nhiếp ảnh số, Linear Regression có thể được sử dụng để xác định bộ lọc tối ưu để loại bỏ nhiễu trong ảnh số. Đồng thời, Linear Regression có thể được sử dụng để phối hợp độ tương phản và độ sáng của ảnh, giúp tăng tính tinh tế cho hình ảnh. Trong việc phân tích video, Linear Regression có thể được sử dụng để dự đoán các giá trị khung hình tiếp theo của video. Việc dự đoán giá trị này có thể giảm độ trễ, ổn định và tăng tốc độ của video. Linear Regression có thể được sử dụng trong việc giải quyết các vấn đề cụ thể trong nhận diện ảnh và video, nhưng không phải là thuật toán phổ biến nhất hay tiên tiến nhất trong lĩnh vực này. Các thuật toán Deep Learning hiện đại (như các mạng Neural Network) đã đạt được kết quả tốt hơn trong việc phân tích hình ảnh và video.
2.	Decision Tree: 
Là một trong những thuật toán machine learning thông dụng cho các bài toán phân loại và dự đoán. Trong nhận diện ảnh và video, Decision Tree có thể được sử dụng để phân loại các hình ảnh hoặc video thành các lớp khác nhau dựa trên các đặc trưng của chúng. Trong bài toán phân loại các bức ảnh, Decision Tree có thể được sử dụng để xác định các đặc trưng như màu sắc, độ tương phản, kích thước vật thể, vị trí vật thể và hướng chụp. Dựa trên các đặc trưng này, thuật toán Decision Tree có thể xác định lớp của ảnh và phân loại nó vào các danh mục tương ứng. Tương tự, trong bài toán phân tích video, Decision Tree có thể được sử dụng để phân loại các video thành các lớp khác nhau dựa trên các đặc trưng như độ phân giải, độ dài, tốc độ khung hình, nội dung video và thời lượng. Thuật toán Decision Tree có thể giúp tự động hóa quá trình phân loại video và giảm thiểu sự can thiệp của con người trong quá trình xử lý và phân loại video. Tuy nhiên, Decision Tree có thể gặp phải một số vấn đề trong việc xử lý những dữ liệu lớn hoặc có tính phức tạp. Vì vậy, trong nhận diện ảnh và video, Decision Tree thường được áp dụng cùng với các thuật toán khác như Random Forest hay Gradient Boosting để giải quyết các vấn đề phức tạp hơn. Decision Tree là một trong những thuật toán machine learning thông dụng và có thể được áp dụng trong việc phân loại và dự đoán các hình ảnh và video. Tuy nhiên, việc áp dụng các thuật toán khác cùng với Decision Tree có thể cải thiện độ chính xác và giúp khắc phục các vấn đề phức tạp trong nhận diện ảnh và video.
3.	KMeans Clustering: 
Là một trong những thuật toán unsupervised learning phổ biến trong machine learning, được sử dụng để phân nhóm dữ liệu thành các nhóm khác nhau dựa trên các đặc trưng của chúng. Trong nhận diện ảnh và video, KMeans Clustering có thể được sử dụng để phân loại các hình ảnh hoặc video thành các cụm khác nhau dựa trên các đặc trưng của chúng. Trong bài toán phân cụm hình ảnh, KMeans Clustering có thể được sử dụng để phân loại các hình ảnh thành các cụm khác nhau dựa trên các đặc trưng như màu sắc, độ tương phản, kích thước vật thể và đặc trưng hình dáng của vật thể. Sau khi phân loại thành các cụm khác nhau, việc xử lý và phân tích được thực hiện cho mỗi cụm đơn lẻ, từ đó giúp ta dễ dàng tìm ra các đặc trưng riêng biệt cho từng nhóm. Tương tự, trong bài toán phân cụm video, KMeans Clustering có thể được sử dụng để phân loại các video thành các cụm khác nhau dựa trên các đặc trưng như độ phân giải, độ dài, tốc độ khung hình, nội dung video và thời lượng. Việc phân nhóm video giúp cho việc phân tích, xử lý và quản lý video trở nên dễ dàng và thuận tiện hơn. Tuy nhiên, KMeans Clustering cũng có thể gặp phải những vấn đề như độ phân cụm không tốt, hoặc khó xác định số lượng phân cụm cần thiết. Do đó, các thuật toán phân cụm hiện đại hơn như DBSCAN hoặc HAC sẽ được sử dụng để cải thiện độ chính xác và hiệu quả của phân loại. KMeans Clustering là một trong những thuật toán unsupervised learning phổ biến và có thể được áp dụng trong việc phân nhóm và phân cụm các hình ảnh và video. Tuy nhiên, việc kết hợp với các thuật toán khác như DBSCAN hoặc HAC có thể cải thiện độ chính xác và hiệu quả của quá trình phân loại.
4.	Support Vector Machine (SVM): 
Là một trong những thuật toán phân loại phổ biến trong machine learning, được sử dụng để phân loại dữ liệu thành các lớp khác nhau dựa trên các đặc trưng. SVM cũng có thể được sử dụng để giải quyết bài toán phân loại trong nhận diện ảnh và video. Trong bài toán phân loại hình ảnh, SVM có thể được sử dụng để phân loại các hình ảnh thành các danh mục khác nhau dựa trên các đặc trưng như màu sắc, độ tương phản, kích thước vật thể và đặc trưng hình dáng. Thuật toán SVM sẽ xác định các đường biên của các nhóm và phân loại phần còn lại của dữ liệu vào danh mục phù hợp. Tương tự, trong bài toán phân tích video, SVM có thể được sử dụng để phân loại các video thành các danh mục khác nhau dựa trên các đặc trưng như độ phân giải, độ dài, tốc độ khung hình, nội dung video và thời lượng. Thuật toán SVM sẽ xác định các đường biên giữa các nhóm và phân loại các video vào các danh mục tương ứng. Tuy nhiên, SVM cũng có thể gặp phải những vấn đề như khó xử lý với dữ liệu lớn, hoặc khó xác định các siêu tham số phù hợp. Do đó, việc lựa chọn đặc trưng đúng cho bài toán cũng như tinh chỉnh các siêu tham số là rất quan trọng để đạt được độ chính xác cao nhất. SVM là một trong những thuật toán phân loại phổ biến trong machine learning và có thể được áp dụng để phân loại các hình ảnh và video. Tuy nhiên, việc lựa chọn đặc trưng đúng và tùy chỉnh siêu tham số là rất quan trọng trong việc đạt được độ chính xác cao nhất.
5.	Random Forest: 
Là một trong những thuật toán phân loại quan trọng trong machine learning, được sử dụng để xác định các đặc trưng quan trọng và cải thiện độ chính xác của mô hình. Trong nhận diện ảnh và video, Random Forest cũng được sử dụng để phân loại các đối tượng trên hình ảnh hoặc video. Trong bài toán phân loại hình ảnh, Random Forest có thể được sử dụng để xác định các đặc trưng quan trọng của hình ảnh và phân loại hình ảnh vào các nhóm khác nhau dựa trên các đặc trưng này. Thuật toán Random Forest hoạt động bằng cách xây dựng nhiều cây quyết định (decision tree) với các tập dữ liệu con được chọn ngẫu nhiên từ tập dữ liệu huấn luyện ban đầu. Mỗi cây quyết định sẽ có các quyết định riêng về các đặc trưng quan trọng và dự đoán nhãn cho mỗi hình ảnh. Tương tự, trong bài toán phân loại video, Random Forest có thể được sử dụng để xác định các đặc trưng quan trọng của video và phân loại chúng vào các danh mục khác nhau. Đối với video, tập dữ liệu có thể được xây dựng bằng cách sử dụng các khung hình (frames) trong video. Tuy nhiên, thuật toán Random Forest cũng có thể có những hạn chế như khó hiểu và khó diễn giải các kết quả dự đoán. Do đó, việc kết hợp với các thuật toán khác như SVM hoặc Neural Networks có thể giúp cải thiện độ chính xác và hiệu quả của phân loại hình ảnh và video. Random Forest là một trong những thuật toán phân loại quan trọng trong machine learning và có thể được sử dụng để phân loại các hình ảnh và video. Tuy nhiên, việc kết hợp với các thuật toán khác có thể cải thiện độ chính xác và hiệu quả của phân loại.
6.	Logistic Regression: 
Là một thuật toán phân loại thường được sử dụng trong machine learning. Nó giúp dự đoán đầu ra dựa trên các biến đầu vào và là một trong những thuật toán phân loại tuyến tính đơn giản nhất. Tuy nhiên, Logistic Regression còn có thể áp dụng được trong bài toán nhận diện ảnh và video. Trong nhận diện ảnh và video, Logistic Regression có thể được áp dụng để phân loại các vật thể và đối tượng trên hình ảnh hoặc video. Việc xác định vật thể trong hình ảnh hoặc video là một trong những bài toán quan trọng trong lĩnh vực Computer Vision. Để thực hiện việc xác định này, các đặc trưng của vật thể trong hình ảnh hoặc video cần được xác định. Logistic Regression có thể được sử dụng để phân loại các đặc trưng này và giúp đưa ra dự đoán về xác suất vật thể được tìm thấy trên hình ảnh hoặc video. Các nhà nghiên cứu đã sử dụng Logistic Regression để phân loại các đối tượng trong hình ảnh, chẳng hạn như việc phân loại đối tượng là xe hơi hoặc xe tải, là người hoặc chó, là cảnh quan hoặc thú vật, v.v. Tuy nhiên, trong nhận diện ảnh và video, Logistic Regression thường được sử dụng kết hợp với các thuật toán khác như Convolutional Neural Networks (CNNs) hoặc Random Forest để đạt được kết quả tốt hơn. Logistic Regression là một thuật toán phân loại được sử dụng phổ biến trong machine learning và cũng có thể được áp dụng trong bài toán nhận diện ảnh và video để phân loại các đối tượng. Tuy nhiên, việc kết hợp với các thuật toán khác có thể giúp cải thiện kết quả chính xác của mô hình.
7.	Convolutional Neural Network (CNN) 
Là một kiểu mạng nơ-ron tích chập (convolutional neural network) được thiết kế để học và trích xuất các đặc trưng của ảnh và video. Việc áp dụng CNN cho việc nhận diện ảnh và video đã giúp nâng cao độ chính xác của việc phân loại và định vị đối tượng.
Quá trình nhận diện ảnh và video bằng CNN được tổng quan là:
Tiền xử lý dữ liệu: Tương tự như các thuật toán khác, việc tiền xử lý dữ liệu là rất quan trọng, bao gồm thay đổi kích thước, giảm nhiễu, chuẩn hóa và chuyển đổi ảnh thành các ma trận.
Tầng tích chập: Sử dụng các bộ lọc (filter) cho việc trích xuất các đặc trưng của ảnh, ví dụ như viền, góc, biên và các đường cong.
Tầng max-pooling: Dùng để giảm kích thước của dữ liệu đầu ra và cải thiện tính bền vững của model.
Tầng dày đặc: Làm phẳng các ma trận đầu ra từ các tầng trước đó thành một vector và kết nối với tầng fully-connected layer.
Tầng fully-connected layer: Tầng kết nối đầy đủ được sử dụng để kết nối các đặc trưng đầu vào với nhãn đầu ra.
Hàm softmax: Đưa ra xác suất của các lớp đầu ra.
Output: Trả về nhãn dự đoán của ảnh hoặc video, và vị trí định vị của đối tượng (nếu có).
Các thuật toán CNN đang được sử dụng rộng rãi trong nhận diện ảnh và video như: YOLO (You Only Look Once), Mask R-CNN, Faster R-CNN và SSD (Single Shot Multibox Detector). Việc áp dụng thành công các thuật toán này làm cho việc phân loại và nhận diện đối tượng trên ảnh và video trở nên chính xác hơn và nhanh hơn, và đồng thời có thể được áp dụng trong nhiều lĩnh vực khác như xe tự hành, an ninh và y tế.
8.	YOLO (You Only Look Once)
 Là một trong những thuật toán được sử dụng để giải quyết vấn đề nhận diện đối tượng trong ảnh và video. Khác với các thuật toán cũ, YOLO không yêu cầu một quá trình phát hiện đối tượng trước đó, mà thực hiện việc phân loại và định vị đối tượng trong một lần duy nhất (one-shot). Với tính năng này, YOLO đạt được độ chính xác cao và tốc độ xử lý nhanh.
Quy trình nhận diện đối tượng trong ảnh và video bằng thuật toán YOLO tường minh như sau:
Tiền xử lý dữ liệu: Tương tự như các thuật toán khác, việc tiền xử lý dữ liệu là rất quan trọng, bao gồm thay đổi kích thước, giảm nhiễu, chuẩn hóa và chuyển đổi ảnh thành các ma trận.
Tổng quan: Ảnh được tổng quan thành một lưới các ô vuông (grid cell) và mỗi ô vuông đại diện cho một khu vực trên ảnh.
Phân lớp: Mỗi ô vuông sẽ được dán nhãn với một lớp đối tượng dự đoán (class object), chẳng hạn như người, xe hơi, chó, mèo,...
Hộp giới hạn: Với mỗi ô vuông, YOLO dự đoán một hoặc nhiều hộp giới hạn (bounding box) để xác định vị trí và hình dạng của đối tượng.
Xác suất: Ngoài việc dự đoán lớp và hộp giới hạn, YOLO cũng đưa ra xác suất dự đoán của mỗi đối tượng trong hộp giới hạn đó.
Loại bỏ: Các hộp giới hạn và đối tượng có xác suất dự đoán thấp hơn một ngưỡng (threshold) được loại bỏ.
Vẽ hộp giới hạn: Cuối cùng, hộp giới hạn của các đối tượng được vẽ trên ảnh gốc để hiển thị kết quả.
Thuật toán YOLO được sử dụng rộng rãi trong việc giải quyết các bài toán phân loại, định vị và nhận dạng đối tượng trong hình ảnh và video. Nó là một trong những thuật toán đáng tin cậy có thể hỗ trợ cho việc giải quyết các vấn đề trên.
Tất cả những ứng dụng này đã mang lại tiềm năng lớn trong việc cải thiện độ chính xác và độ tin cậy của các mô hình nhận diện ảnh và video. Tuy nhiên, vấn đề về tính toán và tốc độ xử lý vẫn cần được cải tiến để thuật toán có thể được tích hợp vào các ứng dụng thực tế.
II. Áp Dụng Của Các Thuật Toán:
1.	Linear Regression:
•	Linear Regression là một phương pháp học có giám sát trong machine learning, được sử dụng để dự đoán giá trị của một biến số dựa trên các biến số độc lập.
•	Trong Linear Regression, chúng ta tìm một mối tương quan tuyến tính giữa các biến độc lập và biến phụ thuộc. Mối tương quan này được biểu diễn bằng một đường thẳng có dạng y = mx + b, trong đó x là biến độc lập, y là biến phụ thuộc, m là hệ số góc của đường thẳng (slope), và b là hệ số chặn (intercept).
•	Các bước cơ bản của Linear Regression là:
+	Chuẩn bị dữ liệu và chia dữ liệu thành tập train và tập test.
+	Xác định 1 hàm mất mát (loss function) và tối ưu hàm này để tìm đường thẳng tốt nhất.
•	Một trong các đường link thể hiện cách áp dụng Linear Regression trong Python bằng thư viện Scikit-learn (1)
2.	Decision Tree:
•	Decision Tree là một thuật toán học có giám sát trong machine learning, được sử dụng để xây dựng một cây quyết định để phân loại hoặc dự đoán dữ liệu.
•	Một Decision Tree được xây dựng dựa trên các quyết định tại các nút của cây. Các nút này có thể là nút gốc, nút nội bộ hoặc nút lá. Mỗi nút sẽ liên kết với một hay nhiều nút con, phụ thuộc vào quyết định được đưa ra tại nút đó.
•	Các bước cơ bản của Decision Tree là:
+	Xây dựng cây quyết định bằng cách chia các dữ liệu thành các nhóm nhỏ hơn.
+	Chọn thuộc tính tốt nhất để chia các nhóm dữ liệu.
+	Lặp lại việc này cho đến khi các điều kiện dừng được đáp ứng.
•	Một trong các đường link thể hiện cách áp dụng Decision Tree trong Python bằng thư viện Scikit-learn (2) 
3.	KMeans Clustering:
•	KMeans Clustering là một thuật toán học không giám sát trong machine learning, được sử dụng để phân nhóm các điểm dữ liệu dựa trên các đặc trưng tương tự.
•	KMeans Clustering phân chia các điểm dữ liệu thành các nhóm dựa trên khoảng cách giữa các điểm. Mỗi điểm sẽ được gán vào nhóm có trung tâm gần nhất.
•	Các bước cơ bản của KMeans Clustering là:
+	Chọn số lượng nhóm.
+	Tìm các trung tâm của các nhóm ban đầu.
+	Gán các điểm dữ liệu vào nhóm tương ứng với trung tâm gần nhất.
+	Cập nhật các trung tâm thông qua việc tính toán trung bình của các điểm trong nhóm.
+	Lặp lại đến khi trung tâm của các nhóm không thay đổi nhiều.
•	Một trong các đường link thể hiện cách áp dụng KMeans Clustering trong Python bằng thư viện Scikit-learn (3)
4.	Support Vector Machine:
•	Support Vector Machine (SVM) là một thuật toán học có giám sát trong machine learning, được sử dụng để phân loại dữ liệu.
•	SVM tìm một đường phân chia giữa các điểm dữ liệu và tối đa hoá khoảng cách từ các điểm dữ liệu tới đường phân chia đó.
•	Các bước cơ bản của SVM là:
+	Tìm đường phân chia tốt nhất bằng cách tối đa hoá khoảng cách giữa các điểm và đường phân chia.
+	Tìm các vectơ hỗ trợ để định vị đường phân chia.
•	Một trong các đường link thể hiện cách sử dụng SVM trong Python bằng thư viện Scikit-learn (4) 
5. Thuật toán Random Forest:
 là một loại thuật toán Ensemble Learning. Nó sử dụng nhiều Decision tree để dự đoán nhãn chính xác cho dữ liệu đầu vào.
Random Forest là một tập hợp các cây quyết định (decision tree) được tạo ra bằng cách chọn ngẫu nhiên một số lượng các đặc trưng và tập con của dữ liệu huấn luyện. Mỗi cây quyết định đưa ra một dự đoán riêng về nhãn của dữ liệu đầu vào. Kết quả dự đoán của Random Forest được tính toán bằng cách lấy trung bình các dự đoán của tất cả các cây quyết định.
Ví dụ, để phân loại các loại trái cây trong hình ảnh, Random Forest có thể sử dụng một tập hợp các cây quyết định để xác định các đặc trưng quan trọng trong hình ảnh và phân loại chúng vào các nhóm khác nhau.
6.	Thuật toán Logistic Regression:
là một thuật toán học có giám sát thường được sử dụng trong các vấn đề phân loại (classification) khi đầu ra là một biến nhị phân (binary) hoặc đa trị (multiclass). Có thể áp dụng cho cả các trường hợp dữ liệu phân loại tuyến tính hoặc phi tuyến tính.
Thuật toán này có khả năng tìm ra quan hệ giữa các biến đầu vào và đầu ra, dựa trên một hàm sigmoid. Hàm sigmoid có giá trị từ 0 đến 1 và có ý nghĩa giải thích xác suất của một phân loại.
Các bước thực hiện Logistic Regression được tóm tắt như sau:
+Tiền xử lý dữ liệu: xử lý các giá trị thiếu (missing values), chuẩn hóa dữ liệu (scaling), chọn đặc trưng (feature selection), và số hóa các biến đầu vào (feature encoding).
+Khởi tạo các trọng số (biases) cho mô hình và tính toán các giá trị đầu ra dựa trên hàm sigmoid.
+Tính toán gradient và đạo hàm riêng của hàm mất mát (loss function) để cập nhật các trọng số cho mô hình.
+Sử dụng thuật toán tối ưu hoá (optimization algorithm) để tối thiểu hóa hàm mất mát và cập nhật trọng số liên tục cho đến khi giá trị hàm mất mát được giảm đến một mức chấp nhận được.
+Kiểm tra hiệu quả của mô hình.
Thuật toán Logistic Regression có ưu điểm là dễ hiểu và áp dụng, cân bằng giữa hiệu quả và độ chính xác, và là một trong những giải pháp đầu tiên được áp dụng để giải quyết vấn đề phân loại. 
7.	Thuật toán YOLO (You Only Look Once):
là một thuật toán dùng cho việc nhận diện đối tượng (object detection) trong hình ảnh (image) và video. YOLO có thể đưa ra dự đoán nhanh chóng và chính xác về vị trí và lớp của đối tượng trên toàn bức ảnh một cách đồng thời.
Các bước thực hiện thuật toán YOLO được tóm tắt như sau:
Input image: Thuật toán YOLO sử dụng một bức ảnh RGB (Red-Green-Blue) với kích thước cố định là 416x416.
Neural Network: Đầu vào bức ảnh sẽ trải qua một mạng nơ-ron tích chập (convolutional neural network) để trích xuất đặc trưng.
Grid cells: Bức ảnh được chia thành một lưới các ô vuông có kích thước cố định. Mỗi ô vuông sẽ dự đoán một số lượng hình chữ nhật (bounding box) và nhãn (class label).
Bounding boxes: Mỗi ô vuông sẽ đưa ra một số lượng hình chữ nhật và mỗi hình chữ nhật đó sẽ đại diện cho một đối tượng. Trong YOLO, mỗi hình chữ nhật sẽ được cân bằng để dự đoán các đối tượng khác nhau hoặc cùng loại.
Confidence score: Mỗi hình chữ nhật sẽ có một điểm tự tin (confidence score) để đánh giá mức độ chính xác của dự đoán về đối tượng đó.
Thuật toán YOLO có ưu điểm là tốc độ xử lý nhanh, chính xác và có thể phát hiện được nhiều đối tượng trong một bức ảnh. 
8.	Thuật toán Convolutional Neural Network (CNN):
là một trong những thuật toán sử dụng trong lĩnh vực xử lý ảnh và thị giác máy tính. CNN đã đạt được những thành công đáng kể trong việc giải quyết các bài toán phân loại ảnh và định vị đối tượng và trở thành công cụ hữu ích trong nhiều lĩnh vực từ phát hiện giả mạo hình ảnh đến tự động lái xe.
Các bước thực hiện của thuật toán CNN:
+ Input image: đầu vào của thuật toán là một hình ảnh kích thước tùy ý.
+ Convolutional layer: Lớp tích chập cơ bản của thuật toán. Giúp trích xuất ra các đặc trưng từ ảnh bằng cách áp dụng các bộ lọc (filter) lên toàn bộ ảnh.
Pooling layer: Nhằm giảm chiều dữ liệu trong quá trình tính toán và giúp giảm ảnh ảnh hưởng của nhiễu và tăng tính tổng quát của mô hình.
Activation function layer: Thường được sử dụng là hàm ReLU để giữ lại các giá trị đặc trưng của các hình ảnh.
Fully connected layer: Tầng kết nối đầy đủ được sử dụng để kết nối các đặc trưng với nhãn đầu ra.
Output: Đầu ra của thuật toán là xác suất của các lớp đầu ra, đường biên hoặc đối tượng dự đoán trong ảnh.
Thuật toán CNN có ưu điểm là có khả năng trích xuất được các đặc trưng tự động từ ảnh, khả năng xử lý ảnh với tốc độ cao và kết quả phân loại chính xác. 
