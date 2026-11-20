import { useDeferredValue, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, Search } from "lucide-react";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, PageHeader, Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { TextField } from "@/components/ui/field";
import { useApiError } from "@/hooks/use-api-error";
import { useCurrentUser } from "@/hooks/use-auth";
import { useFormat } from "@/hooks/use-format";
import { fieldOf } from "@/lib/account-errors";
import { api, type Role, type User } from "@/lib/api";
import { firstError, validateEmail, validateName, validateUsername } from "@/lib/validation";
import { useToastStore } from "@/store/toast-store";

const PAGE = 15;

function UserDialog({ user, self, onClose }: { user: User; self: boolean; onClose: () => void }) {
  const { t } = useTranslation();
  const format = useFormat();
  const client = useQueryClient();
  const describe = useApiError();
  const push = useToastStore((state) => state.push);
  const [form, setForm] = useState({ firstName: user.first_name, lastName: user.last_name, email: user.email ?? "", username: user.username, role: user.role });
  const [temporary, setTemporary] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const errors = { firstName: validateName(form.firstName), lastName: validateName(form.lastName), email: validateEmail(form.email), username: validateUsername(form.username) };
  const settle = () => void client.invalidateQueries({ queryKey: ["users"] });
  const onError = (error: Error) => push("error", describe(error));
  const save = useMutation({ mutationFn: () => api.updateUser(user.id, { firstName: form.firstName, lastName: form.lastName, email: form.email, username: form.username, role: form.role }), onSuccess: () => { settle(); push("success", t("users.saved")); onClose(); }, onError });
  const toggle = useMutation({ mutationFn: () => api.updateUser(user.id, { isActive: !user.is_active }), onSuccess: () => { settle(); onClose(); }, onError });
  const reset = useMutation({ mutationFn: () => api.resetUserPassword(user.id), onSuccess: (result) => { setTemporary(result.temporary_password); settle(); }, onError });
  const revoke = useMutation({ mutationFn: () => api.revokeUserSessions(user.id), onSuccess: (result) => push("success", t("users.revoked", { count: result.revoked })), onError });
  const set = (key: keyof typeof form) => (event: { target: { value: string } }) => setForm((previous) => ({ ...previous, [key]: event.target.value }));
  const fieldError = (key: keyof typeof errors) => (errors[key] ? t(`validation.${errors[key]}`) : fieldOf(save.error) === key ? describe(save.error) : null);
  return (
    <Dialog open title={`${user.first_name} ${user.last_name}`.trim() || user.username} description={`@${user.username}`} onClose={onClose} className="max-w-xl" footer={<><Button variant="outline" onClick={onClose}>{t("common.cancel")}</Button><Button disabled={firstError(errors) || save.isPending} onClick={() => save.mutate()}>{t("common.save")}</Button></>}>
      <div className="grid gap-5">
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Stat label={t("users.status")} value={<StatusBadge value={user.is_active ? "active" : "stopped"} />} />
          <Stat label={t("users.registered")} value={format.dateTime(user.created_at)} />
          <Stat label={t("users.lastActive")} value={user.last_active_at ? format.relative(user.last_active_at) : "—"} hint={user.last_login_at ? `${t("users.lastLogin")}: ${format.dateTime(user.last_login_at)}` : undefined} />
        </dl>
        <div className="grid gap-3">
          <div className="grid grid-cols-2 gap-3">
            <TextField label={t("auth.firstName")} value={form.firstName} onChange={set("firstName")} error={fieldError("firstName")} />
            <TextField label={t("auth.lastName")} value={form.lastName} onChange={set("lastName")} error={fieldError("lastName")} />
          </div>
          <TextField label={t("auth.email")} type="email" value={form.email} onChange={set("email")} error={fieldError("email")} />
          <TextField label={t("auth.username")} value={form.username} onChange={set("username")} error={fieldError("username")} />
          <label className="grid gap-1.5 text-[13px] font-medium">{t("users.role")}
            <select className="field h-9" value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value as Role })}>
              <option value="USER">{t("role.account.USER")}</option>
              <option value="ADMIN">{t("role.account.ADMIN")}</option>
            </select>
          </label>
          {save.error && !fieldOf(save.error) ? <p role="alert" className="text-[13px] text-destructive">{describe(save.error)}</p> : null}
        </div>
        <div className="flex flex-wrap gap-2 border-t hairline pt-4">
          <Button size="sm" variant="outline" disabled={toggle.isPending} onClick={() => toggle.mutate()}>{user.is_active ? t("users.block") : t("users.unblock")}</Button>
          <Button size="sm" variant="outline" disabled={reset.isPending} onClick={() => reset.mutate()}>{t("users.resetPassword")}</Button>
          <Button size="sm" variant="outline" disabled={revoke.isPending || self} onClick={() => revoke.mutate()}>{t("users.revokeSessions")}</Button>
        </div>
        {temporary ? (
          <div role="status" className="grid gap-1.5 rounded-lg bg-muted p-3">
            <p className="text-xs text-muted-foreground">{t("users.temporaryPassword")}</p>
            <div className="flex items-center justify-between gap-2">
              <code className="break-all font-mono text-[13px]">{temporary}</code>
              <Button size="icon" variant="ghost" aria-label={t("common.copy")} onClick={() => { void navigator.clipboard.writeText(temporary); setCopied(true); }}>{copied ? <Check size={14} /> : <Copy size={14} />}</Button>
            </div>
            <p className="text-xs text-muted-foreground">{t("users.temporaryHint")}</p>
          </div>
        ) : null}
      </div>
    </Dialog>
  );
}

export function UsersPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const me = useCurrentUser();
  const [search, setSearch] = useState("");
  const deferred = useDeferredValue(search);
  const [role, setRole] = useState<Role | "">("");
  const [status, setStatus] = useState<"" | "active" | "blocked">("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<User | null>(null);
  const query = useMemo(() => ({ query: deferred, role: role || null, active: status === "" ? null : status === "active", limit: PAGE, offset: page * PAGE }), [deferred, role, status, page]);
  const users = useQuery({ queryKey: ["users", query], queryFn: () => api.users(query), staleTime: 15_000, placeholderData: (previous) => previous });
  const total = users.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  const reset = () => setPage(0);
  return (
    <div className="grid gap-5">
      <PageHeader title={t("users.title")} subtitle={t("users.subtitle", { count: total })} />
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-52 flex-1 sm:max-w-xs">
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input type="search" aria-label={t("users.search")} placeholder={t("users.search")} className="field pl-8" value={search} onChange={(event) => { setSearch(event.target.value); reset(); }} />
        </div>
        <select aria-label={t("users.role")} className="field w-auto" value={role} onChange={(event) => { setRole(event.target.value as Role | ""); reset(); }}>
          <option value="">{t("users.allRoles")}</option>
          <option value="ADMIN">{t("role.account.ADMIN")}</option>
          <option value="USER">{t("role.account.USER")}</option>
        </select>
        <select aria-label={t("users.status")} className="field w-auto" value={status} onChange={(event) => { setStatus(event.target.value as typeof status); reset(); }}>
          <option value="">{t("users.allStatuses")}</option>
          <option value="active">{t("status.active")}</option>
          <option value="blocked">{t("users.blocked")}</option>
        </select>
      </div>
      {users.isPending ? <div role="status" aria-busy="true" className="grid gap-2">{[0, 1, 2, 3].map((item) => <div key={item} className="h-11 animate-pulse rounded bg-muted" />)}</div>
        : users.isError ? <ErrorState error={users.error} retry={() => void users.refetch()} />
        : users.data.items.length === 0 ? <EmptyState title={t("users.empty")} />
        : (
          <div className="overflow-x-auto border-y hairline">
            <table className="data-table">
              <thead><tr><th>{t("users.user")}</th><th>{t("users.role")}</th><th>{t("users.status")}</th><th>{t("users.registered")}</th><th>{t("users.lastActive")}</th></tr></thead>
              <tbody>
                {users.data.items.map((user) => (
                  <tr key={user.id} className="cursor-pointer hover:bg-muted/40" onClick={() => setSelected(user)}>
                    <td className="max-w-64">
                      <button type="button" className="block max-w-full text-left" onClick={(event) => { event.stopPropagation(); setSelected(user); }}>
                        <span className="block truncate font-medium">{`${user.first_name} ${user.last_name}`.trim() || user.username}</span>
                        <span className="block truncate text-xs text-muted-foreground">@{user.username}{user.email ? ` · ${user.email}` : ""}</span>
                      </button>
                    </td>
                    <td>{t(`role.account.${user.role}`)}</td>
                    <td><StatusBadge value={user.is_active ? "active" : "stopped"} /></td>
                    <td>{format.dateTime(user.created_at)}</td>
                    <td>{user.last_active_at ? format.relative(user.last_active_at) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      {pages > 1 ? (
        <div className="flex items-center justify-between text-[13px] text-muted-foreground">
          <span>{t("common.pageOf", { page: page + 1, pages })}</span>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" disabled={page === 0} onClick={() => setPage(page - 1)}>{t("common.previous")}</Button>
            <Button size="sm" variant="outline" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>{t("common.next")}</Button>
          </div>
        </div>
      ) : null}
      {selected ? <UserDialog key={selected.id} user={selected} self={selected.id === me.id} onClose={() => setSelected(null)} /> : null}
    </div>
  );
}
