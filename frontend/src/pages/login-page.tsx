import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import { useUiStore } from "@/store/ui-store";

export function LoginPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { locale, setLocale } = useUiStore();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const login = useMutation({ mutationFn: () => api.login(username, password), onSuccess: () => navigate("/") });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); login.mutate(); }
  return <main className="grid min-h-screen place-items-center p-4"><div className="absolute right-5 top-5"><select className="rounded-md border bg-card px-3 py-2 text-sm" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")} aria-label={t("settings.language")}><option value="ru">RU</option><option value="en">EN</option></select></div><div className="w-full max-w-md"><div className="mb-7"><div className="grid h-10 w-10 place-items-center rounded-xl bg-primary text-base font-bold text-primary-foreground">A</div><h1 className="mt-5 text-2xl font-semibold tracking-tight">{t("auth.title")}</h1><p className="mt-1 text-sm text-muted-foreground">{t("auth.subtitle")}</p></div><Card><CardContent><form className="grid gap-5" onSubmit={submit}><label className="grid gap-2 text-sm font-medium">{t("auth.login")}<input autoComplete="username" className="h-10 rounded-md border bg-transparent px-3 font-normal" value={username} onChange={(event) => setUsername(event.target.value)} /></label><label className="grid gap-2 text-sm font-medium">{t("auth.password")}<input autoComplete="current-password" type="password" className="h-10 rounded-md border bg-transparent px-3 font-normal" value={password} onChange={(event) => setPassword(event.target.value)} /></label>{login.error ? <p role="alert" className="text-sm text-destructive">{login.error.message}</p> : null}<Button className="h-10" type="submit" disabled={login.isPending}>{login.isPending ? t("auth.signingIn") : <>{t("auth.signIn")}<ArrowRight size={16} /></>}</Button><p className="border-t pt-4 text-xs leading-relaxed text-muted-foreground">{t("auth.initialCredentials")}</p></form></CardContent></Card></div></main>;
}
