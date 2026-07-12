import os
import json
import argparse
import subprocess
import yt_dlp
import google.generativeai as genai
from faster_whisper import WhisperModel

def download_video(url):
    print("📥 Mendownload video dan audio...")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'source_video.mp4',
        'merge_output_format': 'mp4',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'keepvideo': True # Simpan video asli, tapi ambil wav-nya juga untuk di-transkrip
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "source_video.mp4", "source_video.wav"

def transcribe_audio(audio_path):
    print("🤖 Menjalankan AI Whisper (Lokal) untuk transkripsi...")
    # Menggunakan model 'tiny' agar ringan dan cepat di CPU GitHub Actions
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    transcript = ""
    for segment in segments:
        # Format: [0.00 - 5.00] Halo semua selamat datang...
        transcript += f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n"
    
    return transcript

def analyze_with_gemini(transcript, prompt, api_key, max_duration):
    print("🧠 Menganalisa momen viral dengan Google Gemini API...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    system_instruction = f"""
    Kamu adalah asisten editor video. Berikut adalah transkrip video beserta waktunya (dalam detik).
    Tugas pengguna: "{prompt}".
    Tugasmu: Cari bagian paling menarik/viral dari transkrip yang sesuai dengan tugas pengguna.
    Durasi maksimal setiap klip adalah {max_duration} detik. Kamu boleh memilih lebih dari 1 klip jika videonya panjang.
    
    WAJIB balas HANYA dengan format JSON valid seperti array di bawah ini, tanpa teks pengantar apapun (tanpa markdown ```json):
    [
      {{"title": "Nama Klip Singkat", "start": 15.5, "end": 45.0}},
      {{"title": "Klip Lucu", "start": 120.0, "end": 150.0}}
    ]
    """
    
    response = model.generate_content(system_instruction + "\n\nTranskrip:\n" + transcript)
    
    try:
        # Bersihkan response jika Gemini menyelipkan markdown
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
        
        # Potong kilat tanpa render ulang menggunakan FFmpeg (stream copy)
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

    # Ambil Gemini API Key dari Environment Variable (GitHub Secrets)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY tidak ditemukan!")
        exit(1)

    video_file, audio_file = download_video(args.url)
    transcript = transcribe_audio(audio_file)
    
    # Jika transkrip kosong, hentikan
    if not transcript.strip():
        print("❌ Tidak ada suara terdeteksi dalam video.")
        exit(1)
        
    clips = analyze_with_gemini(transcript, args.prompt, api_key, args.duration)
    
    if clips:
        cut_video(video_file, clips)
    else:
        print("❌ AI tidak menemukan momen yang cocok dengan prompt Anda.")
