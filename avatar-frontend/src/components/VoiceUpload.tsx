import { useRef, useState } from "react";
import { VOICE_FORMATS } from "../types";
import { validateVoiceFile } from "../validation";

interface Props {
  onChange: (file: File | null) => void;
}

// Voice sample upload: accepts WAV/MP3/FLAC up to 50MB, shows validation errors.
export function VoiceUpload({ onChange }: Props) {
  const [errors, setErrors] = useState<string[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File | null) {
    if (!file) {
      setFileName(null);
      setErrors([]);
      setUploaded(false);
      onChange(null);
      return;
    }
    const result = validateVoiceFile(file);
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

  const accept = VOICE_FORMATS.map((f) => `.${f}`).join(",");

  return (
    <div className="upload-field">
      <label htmlFor="voice-input">聲音樣本</label>
      <input
        id="voice-input"
        ref={inputRef}
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
