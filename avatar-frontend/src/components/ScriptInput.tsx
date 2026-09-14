import { SCRIPT_MAX_CHARS } from "../types";
import { validateScript } from "../validation";

interface Props {
  value: string;
  onChange: (text: string) => void;
}

// Script textarea with a live character counter (max 10000). Content is
// controlled by the parent so it is preserved across validation errors.
export function ScriptInput({ value, onChange }: Props) {
  const result = validateScript(value);
  const over = value.length > SCRIPT_MAX_CHARS;

  return (
    <div className="upload-field">
      <label htmlFor="script-input">講稿內容</label>
      <textarea
        id="script-input"
        value={value}
        rows={8}
        onChange={(e) => onChange(e.target.value)}
        placeholder="輸入要生成的講稿內容（1 至 10000 字）"
      />
      <div className={`char-counter ${over ? "over-limit" : ""}`}>
        {value.length} / {SCRIPT_MAX_CHARS}
      </div>
      {value.length > 0 && !result.valid && (
        <ul className="field-errors" role="alert">
          {result.errors.map((err, i) => (
            <li key={i}>{err}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
