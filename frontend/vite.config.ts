import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { ServerResponse } from "node:http";

const backendPort = Number(process.env.STUDIO_BACKEND_PORT ?? "8000");
if (!Number.isInteger(backendPort) || backendPort < 1 || backendPort > 65535) {
  throw new Error("STUDIO_BACKEND_PORT must be a TCP port");
}
const backend = `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  plugins: [react()],
  cacheDir: ".vite",
  optimizeDeps: {
    include: ["react", "react-dom", "react-dom/client", "react-router-dom", "@tanstack/react-virtual", "recharts", "react-markdown"],
  },
  resolve: {
    dedupe: ["react", "react-dom"],
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "^/api(?:/|[?]|$)": {
        target: backend,
        ws: true,
        configure(proxy) {
          proxy.on("proxyReqWs", (request, _incoming, socket) => {
            socket.once("close", () => request.destroy());
          });
          proxy.on("error", (_error, _request, response) => {
            if (response instanceof ServerResponse && !response.headersSent && !response.writableEnded) {
              response.writeHead(502, { "Content-Type": "application/json" });
              response.end(JSON.stringify({ detail: "studio_unavailable" }));
            }
          });
        },
      },
      "/v1": { target: backend },
      "/health": { target: backend },
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
