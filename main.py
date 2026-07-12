import os
import json
import argparse
import subprocess
import yt_dlp
from google import genai
from faster_whisper import WhisperModel

def download_video(url):
    print("📥 Mendownload video dan audio...")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'source_video.mp4',
        'merge_output_format': 'mp4',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'keepvideo': True
    }
    
    # Membaca cookies untuk menembus blokir bot YouTube
    if os.path.exists("cookies.txt") and os.path.getsize("cookies.txt") > 0:
        print("🍪 Menggunakan file cookies.txt untuk bypass YouTube...")
        ydl_opts['cookiefile'] = 'cookies.txt'
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "source_video.mp4", "source_video.wav"

def transcribe_audio(audio_path):
    print("🤖 Menjalankan AI Whisper (Lokal) untuk transkripsi...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    transcript = ""
    for segment in segments:
        transcript += f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n"
    return transcript

def analyze_with_gemini(transcript, prompt, api_key, max_duration):
    print("🧠 Menganalisa momen viral dengan Google Gemini API (SDK Baru)...")
    client = genai.Client(api_key=api_key)
    
    system_instruction = f"""
    Kamu adalah asisten editor video. Berikut adalah transkrip video beserta waktunya (dalam detik).
    Tugas pengguna: "{prompt}".
    Tugasmu: Cari bagian paling menarik/viral dari transkrip yang sesuai dengan tugas pengguna.
    Durasi maksimal setiap klip adalah {max_duration} detik.
    
    WAJIB balas HANYA dengan format JSON valid seperti array di bawah ini, tanpa markdown ```json:
    [
      {{"title": "Nama Klip Singkat", "start": 15.5, "end": 45.0}}
    ]
    """
    
    # Menggunakan model 2.5-flash untuk kecepatan & dukungan SDK baru
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=system_instruction + "\n\nTranskrip:\n" + transcript
    )
    
    try:
        result_text = response.text.strip().replace('```json', '').replace('```', '')
        clips = json.loads(result_text)
        return clips
    except Exception as e:
        print("❌ Gagal membaca JSON dari Gemini:", response.text)
        return []

def cut_video(video_path, clips):
    print(f"✂️ Memotong {len(clips)} klip berdasarkan analisa AI...")
    os.makedirs("output_clips", exist_ok=True)
    
    for i, clip in enumerate(clips):
        start = clip['start']
        end = clip['end']
        duration = end - start
        
        safe_title = "".join(c if c.isalnum() else "_" for c in clip['title'])
        output_name = f"output_clips/clip_{i+1}_{safe_title}.mp4"
        
        print(f"🎬 Memproses: {clip['title']} ({start}s - {end}s)")
        
        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start), "-t", str(duration),
            "-c:v", "copy", "-c:a", "copy", output_name
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Tersimpan: {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL Video")
    parser.add_argument("--prompt", required=True, help="Instruksi untuk AI")
    parser.add_argument("--duration", type=int, default=60, help="Max durasi klip")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY tidak ditemukan!")
        exit(1)

    video_file, audio_file = download_video(args.url)
    transcript = transcribe_audio(audio_file)
    
    if not transcript.strip():
        print("❌ Tidak ada suara terdeteksi dalam video.")
        exit(1)
        
    clips = analyze_with_gemini(transcript, args.prompt, api_key, args.duration)
    
    if clips:
        cut_video(video_file, clips)
    else:
        print("❌ AI tidak menemukan momen yang cocok dengan prompt Anda.")
