import { defineConfig } from "vite";

// Keep browser requests on the web origin, including when opened from Home Assistant.
export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: process.env.ARTIFACT_API_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api(?=\/|$)/, ""),
      },
    },
  },
});
