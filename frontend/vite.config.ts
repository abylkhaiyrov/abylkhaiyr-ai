import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { ServerResponse } from "node:http";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "^/api(?:/|[?]|$)": {
        target: "http://127.0.0.1:8000",
        ws: true,
        configure(proxy) {
          proxy.on("error", (_error, _request, response) => {
            if (response instanceof ServerResponse && !response.headersSent && !response.writableEnded) {
              response.writeHead(502, { "Content-Type": "application/json" });
              response.end(JSON.stringify({ detail: "studio_unavailable" }));
            }
          });
        },
      },
      "/v1": { target: "http://127.0.0.1:8000" },
      "/health": { target: "http://127.0.0.1:8000" },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ["recharts"],
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
});
