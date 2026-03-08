# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built as a Final Year Project for BCA at Kakatiya Degree College, affiliated with Kakatiya University.

## 👨‍💻 Team
- Nithin Chowdary 
- Guide: Dr. K. Sravana Kumari

---

## 🚀 Features
- Recommends movies based on content similarity
- Uses **TF-IDF**, **KNN**, and **Cosine Similarity**
- Fetches movie posters via **TMDB API**
- Clean and interactive UI built with **Streamlit**
- Dataset of 134+ movies across diverse genres

---

## 🛠️ Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web UI |
| Scikit-learn | ML algorithms (TF-IDF, KNN) |
| TMDB API | Movie posters & metadata |
| Pandas / NumPy | Data processing |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your TMDB API Key
Create a `.env` file in the root folder:
```
TMDB_API_KEY=your_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
movie-recommendation-system/
│
├── app.py                  # Main Streamlit app
├── recommender.py          # ML recommendation logic
├── movies.csv              # Movie dataset (134+ films)
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to exclude from Git
└── README.md               # Project documentation
```

---

## 📌 How It Works
1. User selects a movie from the dropdown
2. TF-IDF vectorizes movie descriptions/genres
3. Cosine Similarity finds the most similar movies
4. Top 5 recommendations are displayed with posters from TMDB

---

## 📜 License
This project is for academic purposes — Final Year BCA Project, 2025-26.
