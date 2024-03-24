from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Dữ liệu mẫu về các đồng hồ và các thuộc tính của chúng
watch_data = [
    {'name': 'Đồng hồ A', 'description': 'Đồng hồ thể thao chống nước', 'price': 100},
    {'name': 'Đồng hồ B', 'description': 'Đồng hồ sang trọng cho nam', 'price': 150},
    {'name': 'Đồng hồ C', 'description': 'Đồng hồ dây da cho nữ', 'price': 120},
    {'name': 'Đồng hồ D', 'description': 'Đồng hồ đeo tay siêu mỏng', 'price': 200},
    # Thêm dữ liệu đồng hồ khác ở đây
]

# Chuyển đổi dữ liệu thành vector TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
watch_descriptions = [watch['description'] for watch in watch_data]
tfidf_matrix = tfidf.fit_transform(watch_descriptions)

# Tính ma trận tương đồng cosine
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Hàm để gợi ý đồng hồ dựa trên mô tả
def get_recommendations(name, cosine_sim=cosine_sim):
    idx = 0
    for i in range(len(watch_data)):
        if watch_data[i]['name'] == name:
            idx = i
            break

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Lấy 3 đồng hồ có độ tương đồng cao nhất

    return [(watch_data[i[0]]['name'], watch_data[i[0]]['price']) for i in sim_scores]

# Ví dụ về việc gợi ý đồng hồ cho người dùng
recommended_watches = get_recommendations('Đồng hồ A')
for watch in recommended_watches:
    print(watch)
