import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "@/app";
import "@/i18n";
import "@/styles.css";

import { useUiStore } from "@/store/ui-store";

document.documentElement.classList.toggle("dark", useUiStore.getState().theme === "dark");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
