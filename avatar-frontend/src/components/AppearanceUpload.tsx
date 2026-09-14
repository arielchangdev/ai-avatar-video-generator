import { useState } from "react";
import { IMAGE_FORMATS, VIDEO_FORMATS } from "../types";
import { validateAppearanceFile } from "../validation";

interface Props {
  onChange: (file: File | null) => void;
}

// Appearance asset upload: accepts image (JPG/PNG/WEBP) or video (MP4/MOV).
export function AppearanceUpload({ onChange }: Props) {
  const [errors, setErrors] = useState<string[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);

  function handleFile(file: File | null) {
    if (!file) {
      setFileName(null);
      setErrors([]);
      setUploaded(false);
      onChange(null);
      return;
    }
    const result = validateAppearanceFile(file);
    setFileName(file.name);
    setErrors(result.errors);
    if (result.valid) {
      setUploaded(true);
      onChange(file);
    } else {
      setUploaded(false);
      onChange(null);
    }
  }

  const accept = [...IMAGE_FORMATS, ...VIDEO_FORMATS]
    .map((f) => `.${f}`)
    .join(",");

  return (
    <div className="upload-field">
      <label htmlFor="appearance-input">人物外觀（圖片或影片）</label>
      <input
        id="appearance-input"
        type="file"
        accept={accept}
        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
      />
      {fileName && <p className="file-name">{fileName}</p>}
      {uploaded && errors.length === 0 && (
        <p className="upload-ok" role="status">
          已上傳
        </p>
      )}
      {errors.length > 0 && (
        <ul className="field-errors" role="alert">
          {errors.map((err, i) => (
            <li key={i}>{err}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
