import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { AuthLayout } from "@/components/account/auth-layout";
import { Button } from "@/components/ui/button";
import { PasswordField, TextField } from "@/components/ui/field";
import { useApiError } from "@/hooks/use-api-error";
import { api } from "@/lib/api";

export function LoginPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const client = useQueryClient();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const describe = useApiError();
  const login = useMutation({
    mutationFn: () => api.login(identifier, password),
    onSuccess: () => {
      client.removeQueries({ queryKey: ["me"], exact: true });
      const target: unknown = location.state?.returnTo;
      navigate(typeof target === "string" && /^\/(?![/\\])/.test(target) ? target : "/", { replace: true });
    },
  });
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (identifier.trim() && password) login.mutate();
  };
  return (
    <AuthLayout title={t("auth.title")} subtitle={t("auth.subtitle")}>
      <form className="grid gap-4" onSubmit={submit} noValidate>
        {location.state?.sessionExpired ? <p role="status" className="text-[13px] text-muted-foreground">{t("auth.sessionExpired")}</p> : null}
        <TextField label={t("auth.identifier")} autoComplete="username" autoFocus value={identifier} onChange={(event) => setIdentifier(event.target.value)} />
        <PasswordField label={t("auth.password")} autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} />
        {login.error ? <p role="alert" className="text-[13px] text-destructive">{describe(login.error)}</p> : null}
        <Button className="h-9" type="submit" disabled={login.isPending || !identifier.trim() || !password}>{login.isPending ? t("auth.signingIn") : <>{t("auth.signIn")}<ArrowRight size={15} /></>}</Button>
        <p className="text-center text-[13px] text-muted-foreground">{t("auth.noAccount")} <Link className="font-medium text-accent hover:underline" to="/register">{t("auth.register")}</Link></p>
        <p className="border-t hairline pt-4 text-xs leading-relaxed text-muted-foreground">{t("auth.initialCredentials")}</p>
      </form>
    </AuthLayout>
  );
}
