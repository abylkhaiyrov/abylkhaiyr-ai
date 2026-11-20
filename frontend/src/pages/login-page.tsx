import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { useApiError } from "@/hooks/use-api-error";
import { api } from "@/lib/api";
import { ApiError } from "@/lib/http";
import { useUiStore } from "@/store/ui-store";

export function LoginPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const client = useQueryClient();
  const { locale, setLocale } = useUiStore();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const describe = useApiError();
  const login = useMutation({ mutationFn: () => api.login(username, password), onSuccess: () => {
    client.removeQueries({ queryKey: ["me"], exact: true });
    const target: unknown = location.state?.returnTo;
    navigate(typeof target === "string" && /^\/(?![/\\])/.test(target) ? target : "/", { replace: true });
  } });
  const submit = (event: FormEvent<HTMLFormElement>) => { event.preventDefault(); login.mutate(); };
  return (
    <main className="grid min-h-dvh place-items-center p-4">
      <div className="absolute right-4 top-4">
        <select className="h-8 rounded-md bg-transparent px-2 text-xs text-muted-foreground hover:bg-muted" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")} aria-label={t("settings.language")}>
          <option value="ru">RU</option>
          <option value="en">EN</option>
        </select>
      </div>
      <div className="w-full max-w-sm">
        <div className="mb-8">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-accent text-sm font-bold text-accent-foreground">A</div>
          <h1 className="mt-5 text-2xl font-semibold tracking-tight">{t("auth.title")}</h1>
          <p className="mt-1 text-[13px] text-muted-foreground">{t("auth.subtitle")}</p>
        </div>
        <form className="grid gap-4" onSubmit={submit}>
          {location.state?.sessionExpired ? <p role="status" className="text-[13px] text-muted-foreground">{t("auth.sessionExpired")}</p> : null}
          <label className="grid gap-1.5 text-[13px] font-medium">{t("auth.login")}
            <input autoComplete="username" className="field h-9" value={username} onChange={(event) => setUsername(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-[13px] font-medium">{t("auth.password")}
            <input autoComplete="current-password" type="password" className="field h-9" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          {login.error ? <p role="alert" className="text-[13px] text-destructive">{login.error instanceof ApiError && login.error.code === "unauthorized" ? t("auth.invalid") : describe(login.error)}</p> : null}
          <Button className="h-9" type="submit" disabled={login.isPending}>{login.isPending ? t("auth.signingIn") : <>{t("auth.signIn")}<ArrowRight size={15} /></>}</Button>
          <p className="border-t hairline pt-4 text-xs leading-relaxed text-muted-foreground">{t("auth.initialCredentials")}</p>
        </form>
      </div>
    </main>
  );
}
