import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite configuration for the AI Avatar frontend.
// Dev server proxies API + WebSocket traffic to the FastAPI backend so the
// frontend can use relative URLs (/api/v1/...) in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api/v1/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
