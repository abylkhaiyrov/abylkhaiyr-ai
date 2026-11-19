import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useUiStore } from "@/store/ui-store";

export function SettingsPage() {
  const client = useQueryClient();
  const { locale, setLocale } = useUiStore();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const change = useMutation({ mutationFn: () => api.changePassword(currentPassword, newPassword), onSuccess: () => client.clear() });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); change.mutate(); }
  return <div className="grid max-w-2xl gap-5"><div><h1 className="text-xl font-semibold">Settings</h1><p className="text-sm text-muted-foreground">Local-only Studio preferences and account security.</p></div><Card><CardHeader><CardTitle>Language</CardTitle></CardHeader><CardContent><label className="grid max-w-xs gap-1 text-sm">Interface language<select className="h-9 rounded-md border bg-transparent px-2" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")}><option value="ru">Русский</option><option value="en">English</option></select></label></CardContent></Card><Card><CardHeader><CardTitle>Change administrator password</CardTitle></CardHeader><CardContent><form className="grid gap-3" onSubmit={submit}><label className="grid gap-1 text-sm">Current password<input type="password" autoComplete="current-password" className="h-9 rounded-md border bg-transparent px-3" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} /></label><label className="grid gap-1 text-sm">New password<input type="password" autoComplete="new-password" minLength={12} className="h-9 rounded-md border bg-transparent px-3" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} /></label>{change.error ? <p role="alert" className="text-sm text-destructive">{change.error.message}</p> : null}<Button className="w-fit" type="submit" disabled={change.isPending}>Update password</Button></form></CardContent></Card></div>;
}
