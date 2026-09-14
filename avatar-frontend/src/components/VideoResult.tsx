import { useState } from "react";
import { audioDownloadUrl, videoDownloadUrl } from "../api";

interface Props {
  taskId: string;
}

// Video preview player plus separate MP4 / MP3 download links. Shows the
// 24-hour download validity and handles an expired link gracefully.
export function VideoResult({ taskId }: Props) {
  const [expired, setExpired] = useState(false);
  const videoUrl = videoDownloadUrl(taskId);
  const audioUrl = audioDownloadUrl(taskId);

  async function handleDownload(url: string, filename: string) {
    const resp = await fetch(url);
    if (resp.status === 410 || resp.status === 404) {
      setExpired(true);
      return;
    }
    const blob = await resp.blob();
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(objectUrl);
  }

  if (expired) {
    return (
      <div className="video-result expired" role="alert">
        <p>下載連結已過期（有效期限為 24 小時）。請重新生成影片。</p>
      </div>
    );
  }

  return (
    <div className="video-result">
      <h3>生成完成</h3>
      <video
        className="preview"
        src={videoUrl}
        controls
        data-testid="video-preview"
        onError={() => setExpired(true)}
      />
      <div className="downloads">
        <button
          type="button"
          onClick={() => handleDownload(videoUrl, "avatar.mp4")}
        >
          下載影片 (MP4)
        </button>
        <button
          type="button"
          onClick={() => handleDownload(audioUrl, "avatar.mp3")}
        >
          下載音訊 (MP3)
        </button>
      </div>
      <p className="expiry-note">下載連結自完成後 24 小時內有效。</p>
    </div>
  );
}
