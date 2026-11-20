import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { en } from "@/locales/en";
import { ru } from "@/locales/ru";

const read = () => {
  try {
    return localStorage.getItem("abylkhaiyr-language");
  } catch {
    return null;
  }
};

const language = read() === "en" ? "en" : "ru";
document.documentElement.lang = language;

void i18n.use(initReactI18next).init({
  resources: { ru: { translation: ru }, en: { translation: en } },
  lng: language,
  fallbackLng: "ru",
  interpolation: { escapeValue: false },
});

export default i18n;
