import customtkinter as ctk
import threading
import requests
from PIL import Image
import io
from googleapiclient.discovery import build
import concurrent.futures
import webbrowser
import re
import os
import sys
from datetime import datetime, timezone

# =============================================================================
# YEREL YAPAY ZEKA KÜTÜPHANELERİ
# =============================================================================
try:
    from transformers import pipeline
    # Torch'u import ediyoruz, CPU sürümü kurulu olsa bile gereklidir.
    import torch 
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("UYARI: Kütüphaneler eksik.")

# =============================================================================
# AYARLAR
# =============================================================================
API_KEY = "AIzaSyDoKXFsQ2rIeGMLTAYXe3oPmogWmfq4vYo" 

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def resource_path(relative_path):
    """ PyInstaller ile paketlendiğinde dosya yollarını bulmak için gereklidir. """
    try:
        # PyInstaller temp klasörü (_MEIPASS)
        base_path = sys._MEIPASS
    except Exception:
        # Normal çalışma yolu
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class YouTubeSmartApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("TrustTube")
        self.geometry("1100x900")
        
        # --- İKON AYARI (Pencere ve Görev Çubuğu İçin) ---
        try:
            icon_path = resource_path("app_icon.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                print("İkon dosyası bulunamadı, varsayılan ikon kullanılıyor.")
        except Exception as e:
            print(f"İkon yükleme hatası: {e}")
        # -------------------------------------------------
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.sentiment_pipeline = None
        self.is_model_loaded = False
        self.loading_label = None # Yükleme mesajı için referans

        self.create_widgets()
        
        if AI_AVAILABLE:
            threading.Thread(target=self.load_local_model, daemon=True).start()

    def create_widgets(self):
        # 1. Başlık
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, pady=(20, 10), sticky="ew")
        
        self.header = ctk.CTkLabel(self.header_frame, text="YouTube Video Finder", font=("Roboto", 24, "bold"))
        self.header.pack()
        
        self.status_label = ctk.CTkLabel(self.header_frame, text="Loading AI Model...", text_color="orange", font=("Arial", 12))
        self.status_label.pack()

        # 2. Arama Alanı
        self.search_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.search_frame.grid(row=1, column=0, pady=10, padx=20, sticky="ew")

        self.entry = ctk.CTkEntry(self.search_frame, placeholder_text="Search topic", height=40, font=("Arial", 14))
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda event: self.start_analysis())

        self.btn_search = ctk.CTkButton(self.search_frame, text="ANALYZE", command=self.start_analysis, height=40, width=120, state="disabled")
        self.btn_search.pack(side="right")

        # 3. Sonuç Alanı
        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Top 5 Recommendations")
        self.scroll_frame.grid(row=2, column=0, pady=20, padx=20, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=1)

    def load_local_model(self):
        try:
            device = -1 # Varsayılan CPU
            device_name = "CPU"

            print(f"System: {device_name}")
            
            online_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            # Exe içinde gömülü model var mı kontrol et
            local_model_path = resource_path("local_model")

            if os.path.exists(local_model_path):
                print(f"Gömülü model bulundu: {local_model_path}")
                model_to_use = local_model_path
            else:
                print("Gömülü model bulunamadı, internetten indiriliyor...")
                model_to_use = online_model_name
            
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_to_use, device=device)
            self.is_model_loaded = True
            
            self.status_label.configure(text=f"● System Ready ({device_name})", text_color="green")
            self.btn_search.configure(state="normal")
            
        except Exception as e:
            self.status_label.configure(text=f"Hata: Model yüklenemedi - {str(e)}", text_color="red")
            print(f"Model yükleme hatası: {e}")

    # --- AI VE DİĞER FONKSİYONLAR ---
    def calculate_ai_score(self, comments_data):
        if not comments_data or not self.sentiment_pipeline: return 50
        total_score = 0
        analyzed_count = 0
        comments_texts = [c['text'] for c in comments_data]
        comments_to_check = comments_texts[:15] 
        try:
            results = self.sentiment_pipeline(comments_to_check, truncation=True, max_length=512)
            for res in results:
                label = res['label'].lower()
                score = res['score']
                if 'positive' in label: total_score += (100 * score)
                elif 'neutral' in label: total_score += 50
                elif 'negative' in label: total_score += (100 - (score * 100)) * 0.2
                else: total_score += 50
                analyzed_count += 1
            if analyzed_count == 0: return 50
            return int(total_score / analyzed_count)
        except Exception: return 50

    @staticmethod
    def parse_duration(duration_str):
        if not duration_str: return 0
        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration_str)
        if not match: return 0
        hours = int(match.group(1)[:-1]) if match.group(1) else 0
        minutes = int(match.group(2)[:-1]) if match.group(2) else 0
        seconds = int(match.group(3)[:-1]) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def calculate_days_old(published_at_str):
        try:
            pub_date = datetime.strptime(published_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(1, (now - pub_date).days)
        except: return 1

    @staticmethod
    def get_comments(video_id):
        if not API_KEY or "YOUR_API_KEY" in API_KEY: return []
        try:
            youtube = build("youtube", "v3", developerKey=API_KEY)
            resp = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=15, order='relevance').execute()
            comments_data = []
            for item in resp.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                text = snippet.get('textOriginal', snippet.get('textDisplay', ''))
                likes = int(snippet.get('likeCount', 0))
                comments_data.append({'text': text, 'likes': likes})
            return comments_data
        except Exception: return []

    @staticmethod
    def get_batch_stats(video_ids):
        if not API_KEY or "YOUR_API_KEY" in API_KEY: return {}
        try:
            youtube = build("youtube", "v3", developerKey=API_KEY)
            stats = {}
            for i in range(0, len(video_ids), 50):
                chunk = video_ids[i:i+50]
                resp = youtube.videos().list(part='statistics,contentDetails', id=",".join(chunk)).execute()
                for item in resp.get("items", []):
                    data = item["statistics"]
                    data["duration"] = item["contentDetails"]["duration"]
                    stats[item["id"]] = data
            return stats
        except Exception: return {}

    def fetch_and_score_data(self, query, status_callback=None):
        """
        Verileri çeker ve işler.
        status_callback: Arayüzdeki yazıyı güncellemek için fonksiyon.
        """
        if "YOUR_API_KEY" in API_KEY: return [], "API Key Hatası"
        
        # --- ADIM 1: ARAMA ---
        msg = f"Fetching videos for: '{query}'..."
        print(msg)
        if status_callback: status_callback(msg)

        try:
            youtube = build("youtube", "v3", developerKey=API_KEY)
            search_resp = youtube.search().list(q=query, part="snippet", type="video", maxResults=50).execute()
        except Exception as e: return [], f"Hata: {e}"
        
        videos = []
        video_ids = []
        for item in search_resp.get("items", []):
            if "videoId" not in item["id"]: continue
            vid = item["id"]["videoId"]
            video_ids.append(vid)
            videos.append({
                "id": vid, "title": item["snippet"]["title"], "thumb": item["snippet"]["thumbnails"]["medium"]["url"],
                "published_at": item["snippet"]["publishedAt"], "comments": [], "ai_score": 0, "final_score": 0, "stats": {}
            })
        if not videos: return [], "Video bulunamadı."

        # --- ADIM 2: İSTATİSTİK VE SHORTS ---
        msg = "Gathering stats & Filtering Shorts..."
        print(msg)
        if status_callback: status_callback(msg)

        stats_map = self.get_batch_stats(video_ids)
        filtered_videos = []
        for v in videos:
            st = stats_map.get(v["id"], {})
            v["stats"] = st
            duration_sec = self.parse_duration(st.get("duration", "PT0S"))
            if duration_sec > 60: filtered_videos.append(v)
        videos = filtered_videos
        if not videos: return [], "Sadece Shorts bulundu."

        # --- ADIM 3: AI ANALİZİ ---
        msg = "Reading comments & Running AI Analysis..."
        print(msg)
        if status_callback: status_callback(msg)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_video = {executor.submit(self.get_comments, v["id"]): v for v in videos}
            for future in concurrent.futures.as_completed(future_to_video):
                video = future_to_video[future]
                video["comments"] = future.result()
                video["ai_score"] = self.calculate_ai_score(video["comments"])

        # --- ADIM 4: HESAPLAMA ---
        msg = "Finalizing Scores..."
        print(msg)
        if status_callback: status_callback(msg)

        max_daily = 0
        max_eng = 0
        for v in videos:
            views = int(v["stats"].get("viewCount", 0))
            likes = int(v["stats"].get("likeCount", 0))
            daily = views / self.calculate_days_old(v["published_at"])
            eng = (likes / views) if views > 0 else 0
            v["raw_daily"] = daily
            v["raw_eng"] = eng
            if daily > max_daily: max_daily = daily
            if eng > max_eng: max_eng = eng

        for v in videos:
            norm_vel = (v["raw_daily"] / max_daily * 100) if max_daily > 0 else 0
            norm_eng = (v["raw_eng"] / max_eng * 100) if max_eng > 0 else 0
            v["final_score"] = int((v["ai_score"] * 0.60) + (norm_vel * 0.20) + (norm_eng * 0.20))

        return sorted(videos, key=lambda x: x["final_score"], reverse=True)[:5], None

    def download_image(self, url):
        try: return ctk.CTkImage(Image.open(io.BytesIO(requests.get(url, timeout=3).content)), size=(192, 108))
        except: return None

    def start_analysis(self):
        if not self.is_model_loaded: return
        query = self.entry.get()
        if not query: return
        self.btn_search.configure(state="disabled", text="Working...")
        
        # Sonuç ekranını temizle
        for w in self.scroll_frame.winfo_children(): w.destroy()
        
        # Bilgi mesajı için Label oluştur ve referansını sakla
        self.loading_label = ctk.CTkLabel(self.scroll_frame, text="Starting...", font=("Arial", 16))
        self.loading_label.pack(pady=40)

        threading.Thread(target=self.run_process, args=(query,), daemon=True).start()

    def update_loading_status(self, text):
        """Ana thread üzerinden loading yazısını güvenli şekilde günceller."""
        if self.loading_label:
            self.after(0, lambda: self.loading_label.configure(text=text))

    def run_process(self, query):
        # Callback fonksiyonunu parametre olarak gönderiyoruz
        top_5_videos, error_msg = self.fetch_and_score_data(query, status_callback=self.update_loading_status)
        
        if error_msg: self.after(0, self.update_ui_error, error_msg); return
        images = [self.download_image(v["thumb"]) for v in top_5_videos]
        self.after(0, self.update_ui, top_5_videos, images)

    def update_ui_error(self, message):
        for w in self.scroll_frame.winfo_children(): w.destroy()
        ctk.CTkLabel(self.scroll_frame, text="HATA", text_color="red").pack(pady=10)
        ctk.CTkLabel(self.scroll_frame, text=message).pack()
        self.btn_search.configure(state="normal", text="ANALYZE")

    def update_ui(self, videos, images):
        for w in self.scroll_frame.winfo_children(): w.destroy()
        if not videos:
            ctk.CTkLabel(self.scroll_frame, text="No results found.").pack()
            self.btn_search.configure(state="normal", text="ANALYZE"); return

        for i, vid in enumerate(videos):
            score = vid["final_score"]
            score_color = "#1c8d4b" if score > 75 else "#be9b0f" if score > 50 else "#b9392b"
            card = ctk.CTkFrame(self.scroll_frame, fg_color=("gray90", "gray20"))
            card.pack(fill="x", pady=5, padx=5)
            
            ctk.CTkButton(card, text="WATCH ↗", width=80, height=40, command=lambda v=vid["id"]: webbrowser.open(f"https://www.youtube.com/watch?v={v}")).pack(side="right", padx=15, pady=10)
            ctk.CTkLabel(card, text="", image=images[i]).pack(side="left", padx=10, pady=10)
            
            info = ctk.CTkFrame(card, fg_color="transparent")
            info.pack(side="left", fill="both", expand=True, padx=5)
            ctk.CTkLabel(info, text=f"#{i+1} {vid['title']}", font=("Arial", 14, "bold"), anchor="w").pack(fill="x")
            
            sf = ctk.CTkFrame(info, fg_color="transparent")
            sf.pack(fill="x", pady=5)
            ctk.CTkButton(sf, text=f"TOTAL SCORE: {score}", fg_color=score_color, width=130, hover=False).pack(side="left")
            
            stats = vid["stats"]
            views = f"{int(stats.get('viewCount',0)):,}"
            likes = stats.get('likeCount', 0)
            date = vid.get("published_at", "")[:10]
            ctk.CTkLabel(sf, text=f" (AI: {vid['ai_score']}) | 📅 {date} | 👁 {views} | 👍 {likes}", text_color="gray", font=("Arial", 11)).pack(side="left", padx=10)

            if vid["comments"]:
                best = max(vid["comments"], key=lambda x: x['likes'])
                txt = best['text'][:120] + "..." if len(best['text']) > 120 else best['text']
                ctk.CTkLabel(info, text=f"💡 Top Comment ({best['likes']} likes): \"{txt}\"", text_color="silver", font=("Arial", 11, "italic"), anchor="w").pack(fill="x")

        self.btn_search.configure(state="normal", text="ANALYZE")

if __name__ == "__main__":
    app = YouTubeSmartApp()
    app.mainloop()