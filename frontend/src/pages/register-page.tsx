import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { AuthLayout } from "@/components/account/auth-layout";
import { Button } from "@/components/ui/button";
import { PasswordField, TextField } from "@/components/ui/field";
import { useApiError } from "@/hooks/use-api-error";
import { fieldOf } from "@/lib/account-errors";
import { api } from "@/lib/api";
import { firstError, normalizeEmail, normalizeUsername, validateEmail, validateName, validatePassword, validateUsername, type FieldCode } from "@/lib/validation";

type Form = { firstName: string; lastName: string; email: string; username: string; password: string; confirm: string };
const EMPTY: Form = { firstName: "", lastName: "", email: "", username: "", password: "", confirm: "" };

function check(form: Form): Record<keyof Form, FieldCode | null> {
  return {
    firstName: validateName(form.firstName),
    lastName: validateName(form.lastName),
    email: validateEmail(form.email),
    username: validateUsername(form.username),
    password: validatePassword(form.password, form.username),
    confirm: form.confirm === form.password ? null : "passwordMismatch",
  };
}

export function RegisterPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const client = useQueryClient();
  const describe = useApiError();
  const [form, setForm] = useState<Form>(EMPTY);
  const [touched, setTouched] = useState(false);
  const errors = check(form);
  const register = useMutation({
    mutationFn: () => api.register({ firstName: form.firstName.trim(), lastName: form.lastName.trim(), email: normalizeEmail(form.email), username: normalizeUsername(form.username), password: form.password }),
    onSuccess: () => {
      client.removeQueries({ queryKey: ["me"], exact: true });
      navigate("/chat", { replace: true });
    },
  });
  const set = (key: keyof Form) => (event: { target: { value: string } }) => setForm((previous) => ({ ...previous, [key]: event.target.value }));
  const shown = (key: keyof Form) => {
    const code = touched ? errors[key] : null;
    if (code) return t(`validation.${code}`);
    return fieldOf(register.error) === key ? describe(register.error) : null;
  };
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTouched(true);
    if (!firstError(errors)) register.mutate();
  };
  const generalError = register.error && !fieldOf(register.error) ? describe(register.error) : null;
  return (
    <AuthLayout title={t("auth.registerTitle")} subtitle={t("auth.registerSubtitle")}>
      <form className="grid gap-3.5" onSubmit={submit} noValidate>
        <div className="grid grid-cols-2 gap-3">
          <TextField label={t("auth.firstName")} autoComplete="given-name" autoFocus value={form.firstName} onChange={set("firstName")} error={shown("firstName")} />
          <TextField label={t("auth.lastName")} autoComplete="family-name" value={form.lastName} onChange={set("lastName")} error={shown("lastName")} />
        </div>
        <TextField label={t("auth.email")} type="email" autoComplete="email" value={form.email} onChange={set("email")} error={shown("email")} />
        <TextField label={t("auth.username")} autoComplete="username" value={form.username} onChange={set("username")} error={shown("username")} hint={t("auth.usernameHint")} />
        <PasswordField label={t("auth.password")} autoComplete="new-password" value={form.password} onChange={set("password")} error={shown("password")} hint={t("auth.passwordHint")} />
        <PasswordField label={t("auth.confirmPassword")} autoComplete="new-password" value={form.confirm} onChange={set("confirm")} error={shown("confirm")} />
        {generalError ? <p role="alert" className="text-[13px] text-destructive">{generalError}</p> : null}
        <Button className="mt-1 h-9" type="submit" disabled={register.isPending}>{register.isPending ? t("auth.registering") : <>{t("auth.register")}<ArrowRight size={15} /></>}</Button>
        <p className="text-center text-[13px] text-muted-foreground">{t("auth.haveAccount")} <Link className="font-medium text-accent hover:underline" to="/login">{t("auth.signIn")}</Link></p>
      </form>
    </AuthLayout>
  );
}
