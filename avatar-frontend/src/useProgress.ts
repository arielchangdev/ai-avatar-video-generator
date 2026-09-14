import { useEffect, useRef, useState } from "react";
import { getTask, progressSocketUrl } from "./api";
import type { ErrorMessage, TaskProgress, WsMessage } from "./types";

export interface ProgressState {
  progress: TaskProgress | null;
  error: ErrorMessage | null;
  connected: boolean;
}

// Subscribes to a task's progress WebSocket. On mount (and reconnection) it
// fetches the current state via REST so a returning user immediately sees the
// latest progress even before the socket delivers an update.
export function useProgress(taskId: string | null): ProgressState {
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [error, setError] = useState<ErrorMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!taskId) return;

    let cancelled = false;

    // Restore current state on (re)entry.
    getTask(taskId)
      .then((p) => {
        if (!cancelled) setProgress(p);
      })
      .catch(() => {
        /* socket will deliver state shortly */
      });

    const ws = new WebSocket(progressSocketUrl(taskId));
    socketRef.current = ws;

    ws.onopen = () => !cancelled && setConnected(true);
    ws.onclose = () => !cancelled && setConnected(false);
    ws.onmessage = (event) => {
      if (cancelled) return;
      let msg: WsMessage;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === "error") {
        setError(msg);
      } else if (msg.type === "progress") {
        setProgress(msg);
      }
    };

    return () => {
      cancelled = true;
      ws.close();
      socketRef.current = null;
    };
  }, [taskId]);

  return { progress, error, connected };
}
