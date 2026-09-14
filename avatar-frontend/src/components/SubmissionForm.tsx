import { useMemo, useState } from "react";
import { VoiceUpload } from "./VoiceUpload";
import { AppearanceUpload } from "./AppearanceUpload";
import { ScriptInput } from "./ScriptInput";
import { appearanceKind, validateScript } from "../validation";

export interface SubmissionPayload {
  voiceSample: File;
  appearanceAsset: File;
  scriptText: string;
}

interface Props {
  onSubmit: (payload: SubmissionPayload) => void;
  submitting: boolean;
  serverErrors: string[];
}

// Combines the three inputs, shows a material summary, and requires an explicit
// confirmation before generation starts.
export function SubmissionForm({ onSubmit, submitting, serverErrors }: Props) {
  const [voice, setVoice] = useState<File | null>(null);
  const [appearance, setAppearance] = useState<File | null>(null);
  const [script, setScript] = useState("");

  const scriptValid = validateScript(script).valid;

  // Per-item readiness for the confirm gate.
  const missing = useMemo(() => {
    const items: string[] = [];
    if (!voice) items.push("聲音樣本");
    if (!appearance) items.push("人物外觀素材");
    if (!scriptValid) items.push("有效的講稿");
    return items;
  }, [voice, appearance, scriptValid]);

  const ready = missing.length === 0;

  const summary =
    ready && voice && appearance
      ? {
          appearanceType: appearanceKind(appearance),
          scriptChars: script.length,
          voiceName: voice.name,
        }
      : null;

  function handleConfirm() {
    if (ready && voice && appearance) {
      onSubmit({ voiceSample: voice, appearanceAsset: appearance, scriptText: script });
    }
  }

  return (
    <div className="submission-form">
      <VoiceUpload onChange={setVoice} />
      <AppearanceUpload onChange={setAppearance} />
      <ScriptInput value={script} onChange={setScript} />

      {summary && (
        <div className="summary" data-testid="material-summary">
          <h3>素材摘要</h3>
          <ul>
            <li>聲音樣本：{summary.voiceName}</li>
            <li>
              外觀素材類型：{summary.appearanceType === "image" ? "圖片" : "影片"}
            </li>
            <li>講稿字數：{summary.scriptChars} 字</li>
          </ul>
        </div>
      )}

      {!ready && (
        <p className="pending-items">
          尚需提供：{missing.join("、")}
        </p>
      )}

      {serverErrors.length > 0 && (
        <ul className="field-errors" role="alert">
          {serverErrors.map((err, i) => (
            <li key={i}>{err}</li>
          ))}
        </ul>
      )}

      <button
        type="button"
        className="confirm-button"
        disabled={!ready || submitting}
        onClick={handleConfirm}
      >
        {submitting ? "建立中…" : "確認並開始生成"}
      </button>
    </div>
  );
}
