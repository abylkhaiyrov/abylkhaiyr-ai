import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "@/app";
import "@/i18n";
import "@/styles.css";

const savedTheme = localStorage.getItem("abylkhaiyr-theme");
document.documentElement.classList.toggle("dark", savedTheme !== "light");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
