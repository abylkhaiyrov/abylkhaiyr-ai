import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { AlertTriangle } from "lucide-react";
import { PageHeader, Panel } from "@/components/data-state";
import { Button } from "@/components/ui/button";
import { PasswordField, TextField } from "@/components/ui/field";
import { useApiError } from "@/hooks/use-api-error";
import { useCurrentUser } from "@/hooks/use-auth";
import { useFormat } from "@/hooks/use-format";
import { fieldOf } from "@/lib/account-errors";
import { api } from "@/lib/api";
import { firstError, normalizeEmail, normalizeUsername, validateEmail, validateName, validatePassword, validateUsername } from "@/lib/validation";
import { useToastStore } from "@/store/toast-store";

function ProfileForm() {
  const { t } = useTranslation();
  const user = useCurrentUser();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const [form, setForm] = useState({ firstName: user.first_name, lastName: user.last_name, email: user.email ?? "", username: user.username });
  const [touched, setTouched] = useState(false);
  const errors = { firstName: validateName(form.firstName), lastName: validateName(form.lastName), email: validateEmail(form.email), username: validateUsername(form.username) };
  const save = useMutation({
    mutationFn: () => api.updateProfile({ firstName: form.firstName.trim(), lastName: form.lastName.trim(), email: normalizeEmail(form.email), username: normalizeUsername(form.username) }),
    onSuccess: (updated) => {
      client.setQueryData(["me"], updated);
      push("success", t("account.profileSaved"));
    },
  });
  const set = (key: keyof typeof form) => (event: { target: { value: string } }) => setForm((previous) => ({ ...previous, [key]: event.target.value }));
  const shown = (key: keyof typeof form) => (touched && errors[key] ? t(`validation.${errors[key]}`) : fieldOf(save.error) === key ? describe(save.error) : null);
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTouched(true);
    if (!firstError(errors)) save.mutate();
  };
  return (
    <Panel title={t("account.personal")}>
      <form className="grid max-w-lg gap-3.5" onSubmit={submit} noValidate>
        <div className="grid grid-cols-2 gap-3">
          <TextField label={t("auth.firstName")} value={form.firstName} onChange={set("firstName")} error={shown("firstName")} />
          <TextField label={t("auth.lastName")} value={form.lastName} onChange={set("lastName")} error={shown("lastName")} />
        </div>
        <TextField label={t("auth.email")} type="email" value={form.email} onChange={set("email")} error={shown("email")} />
        <TextField label={t("auth.username")} value={form.username} onChange={set("username")} error={shown("username")} />
        {save.error && !fieldOf(save.error) ? <p role="alert" className="text-[13px] text-destructive">{describe(save.error)}</p> : null}
        <Button className="w-fit" type="submit" disabled={save.isPending}>{t("common.save")}</Button>
      </form>
    </Panel>
  );
}

function PasswordForm() {
  const { t } = useTranslation();
  const user = useCurrentUser();
  const client = useQueryClient();
  const navigate = useNavigate();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const [form, setForm] = useState({ current: "", next: "", confirm: "" });
  const [touched, setTouched] = useState(false);
  const errors = { current: form.current ? null : "required", next: validatePassword(form.next, user.username), confirm: form.confirm === form.next ? null : "passwordMismatch" } as const;
  const change = useMutation({
    mutationFn: () => api.changePassword(form.current, form.next),
    onSuccess: () => {
      push("success", t("auth.passwordUpdated"));
      client.clear();
      navigate("/login", { replace: true });
    },
  });
  const set = (key: keyof typeof form) => (event: { target: { value: string } }) => setForm((previous) => ({ ...previous, [key]: event.target.value }));
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTouched(true);
    if (!firstError(errors)) change.mutate();
  };
  const shown = (key: keyof typeof errors) => (touched && errors[key] ? t(`validation.${errors[key]}`) : null);
  return (
    <Panel title={t("auth.changePassword")}>
      <form className="grid max-w-lg gap-3.5" onSubmit={submit} noValidate>
        <PasswordField label={t("auth.currentPassword")} autoComplete="current-password" value={form.current} onChange={set("current")} error={shown("current") ?? (fieldOf(change.error) === "currentPassword" ? describe(change.error) : null)} />
        <PasswordField label={t("auth.newPassword")} autoComplete="new-password" value={form.next} onChange={set("next")} error={shown("next") ?? (fieldOf(change.error) === "password" ? describe(change.error) : null)} hint={t("auth.passwordHint")} />
        <PasswordField label={t("auth.confirmPassword")} autoComplete="new-password" value={form.confirm} onChange={set("confirm")} error={shown("confirm")} />
        {change.error && !fieldOf(change.error) ? <p role="alert" className="text-[13px] text-destructive">{describe(change.error)}</p> : null}
        <Button className="w-fit" type="submit" disabled={change.isPending}>{t("auth.changePassword")}</Button>
      </form>
    </Panel>
  );
}

function Sessions() {
  const { t } = useTranslation();
  const format = useFormat();
  const client = useQueryClient();
  const describe = useApiError();
  const push = useToastStore((state) => state.push);
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessions, staleTime: 30_000 });
  const settle = () => void client.invalidateQueries({ queryKey: ["sessions"] });
  const onError = (error: Error) => push("error", describe(error));
  const revoke = useMutation({ mutationFn: api.revokeSession, onSuccess: settle, onError });
  const revokeOthers = useMutation({ mutationFn: api.revokeOtherSessions, onSuccess: settle, onError });
  const items = sessions.data ?? [];
  return (
    <Panel title={t("account.sessions")} actions={<Button size="sm" variant="outline" disabled={items.length < 2 || revokeOthers.isPending} onClick={() => revokeOthers.mutate()}>{t("account.revokeOthers")}</Button>}>
      {sessions.isPending ? <div className="h-16 animate-pulse rounded-lg bg-muted" /> : sessions.isError ? <p role="alert" className="text-[13px] text-destructive">{describe(sessions.error)}</p> : (
        <ul className="divide-y hairline">
          {items.map((item) => (
            <li key={item.id} className="flex items-center justify-between gap-3 py-2.5">
              <div className="min-w-0">
                <p className="truncate text-[13px] font-medium">{item.user_agent ?? t("account.unknownDevice")}{item.current ? <span className="ml-2 text-xs font-normal text-success">{t("account.currentSession")}</span> : null}</p>
                <p className="text-xs text-muted-foreground">{t("account.signedIn")}: {format.dateTime(item.created_at)} · {t("account.lastSeen")}: {format.relative(item.last_seen_at)}</p>
              </div>
              {item.current ? null : <Button size="sm" variant="ghost" disabled={revoke.isPending} onClick={() => revoke.mutate(item.id)}>{t("account.revoke")}</Button>}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function ProfilePage() {
  const { t } = useTranslation();
  const user = useCurrentUser();
  return (
    <div className="grid gap-8">
      <PageHeader title={t("account.profile")} subtitle={`${user.username} · ${t(`role.account.${user.role}`)}`} />
      {user.requires_password_change ? (
        <div role="alert" className="flex gap-2 text-[13px] text-warning"><AlertTriangle size={15} className="mt-0.5 shrink-0" />{t("account.mustChangePassword")}</div>
      ) : null}
      <PasswordForm />
      <ProfileForm />
      <Sessions />
    </div>
  );
}
