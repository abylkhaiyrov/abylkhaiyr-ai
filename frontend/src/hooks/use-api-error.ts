import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { ApiError } from "@/lib/http";

export function useApiError() {
  const { t, i18n } = useTranslation();
  return useCallback(
    (error: unknown, fallback = "apiError.unknown") => {
      if (!(error instanceof ApiError)) return t(fallback);
      const specific = error.detail ? `apiError.detail.${error.detail}` : null;
      if (specific && i18n.exists(specific)) return t(specific);
      return t(`apiError.${error.code}`);
    },
    [t, i18n],
  );
}
