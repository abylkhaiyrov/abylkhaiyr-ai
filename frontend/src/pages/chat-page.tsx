import { useEffect, useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { Copy, Plus, Send, Square, Trash2, Pencil } from "lucide-react";
import { api } from "@/lib/api";
import { useStudioEvents, type StudioEvent } from "@/hooks/use-studio-events";
import { Button } from "@/components/ui/button";
import { ErrorState, LoadingState } from "@/components/data-state";

type Generation = { requestId: string; chatId: string; messageId?: string; text: string; sequence: number; status: string };

export function ChatPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const { send, subscribe } = useStudioEvents();
  const [selected, setSelected] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [search, setSearch] = useState("");
  const [active, setActive] = useState<Generation | null>(null);
  const [error, setError] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const chats = useQuery({ queryKey: ["chats", search], queryFn: () => api.chats(search) });
  const dashboard = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard });
  const chat = useQuery({ queryKey: ["chat", selected], queryFn: () => api.chat(selected ?? ""), enabled: selected !== null });
  const create = useMutation({ mutationFn: api.createChat, onSuccess: (item) => { void client.invalidateQueries({ queryKey: ["chats"] }); setSelected(item.id); } });
  const remove = useMutation({ mutationFn: api.deleteChat, onSuccess: () => { setSelected(null); void client.invalidateQueries({ queryKey: ["chats"] }); } });
  const rename = useMutation({ mutationFn: ({ id, title }: { id: string; title: string }) => api.renameChat(id, title), onSuccess: () => { void client.invalidateQueries({ queryKey: ["chats"] }); void client.invalidateQueries({ queryKey: ["chat"] }); } });

  useEffect(() => subscribe((event: StudioEvent) => {
    if (!event.type.startsWith("generation.") || !event.request_id) return;
    if (event.type === "generation.started") void client.invalidateQueries({ queryKey: ["chat"] });
    if (event.type === "generation.failed") setError(true);
    if (["generation.completed", "generation.cancelled", "generation.failed"].includes(event.type)) {
      void client.invalidateQueries({ queryKey: ["chat"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
    }
    setActive((previous) => {
      if (event.type === "generation.snapshot" && event.chat_id) {
        if (previous && previous.requestId === event.request_id && (event.sequence_id ?? 0) < previous.sequence) return previous;
        return { requestId: event.request_id!, chatId: event.chat_id, messageId: event.message_id, text: event.text ?? "", sequence: event.sequence_id ?? 0, status: event.status ?? "running" };
      }
      if (!previous || previous.requestId !== event.request_id || (event.sequence_id ?? 0) <= previous.sequence) return previous;
      return { ...previous, messageId: event.message_id ?? previous.messageId, sequence: event.sequence_id ?? previous.sequence, text: event.type === "generation.delta" ? previous.text + (event.text ?? "") : event.text ?? previous.text, status: event.type.slice(11) };
    });
  }), [subscribe, client]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.text, chat.data?.messages.length]);
  const running = active !== null && ["queued", "started", "prefill", "delta", "running"].includes(active.status);
  const visible = active?.chatId === selected ? active : null;
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected || !prompt.trim() || running) return;
    const requestId = crypto.randomUUID();
    if (!send({ type: "generation.start", request_id: requestId, chat_id: selected, content: prompt.trim() })) { setError(true); return; }
    setActive({ requestId, chatId: selected, text: "", sequence: 0, status: "queued" });
    setPrompt("");
    setError(false);
  }
  if (chats.isPending) return <LoadingState />;
  if (chats.isError) return <ErrorState error={chats.error} retry={() => chats.refetch()} />;
  return <div className="grid min-h-[calc(100vh-8rem)] overflow-hidden rounded-xl border bg-card md:grid-cols-[15rem_minmax(0,1fr)]">
    <aside className="border-b p-3 md:border-b-0 md:border-r">
      <Button className="w-full gap-2" onClick={() => create.mutate()} disabled={create.isPending}><Plus size={16} />{t("chat.newChat")}</Button>
      <input className="my-3 w-full rounded-md border bg-background px-3 py-2 text-sm" value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t("chat.searchChats")} aria-label={t("chat.searchChats")} />
      <nav className="grid max-h-64 gap-1 overflow-auto md:max-h-[65vh]" aria-label={t("chat.searchChats")}>{chats.data.map((item) => <button key={item.id} className={`truncate rounded-md px-3 py-2 text-left text-sm ${selected === item.id ? "bg-muted font-medium" : "hover:bg-muted"}`} onClick={() => setSelected(item.id)}>{item.title === "New chat" ? t("chat.newChat") : item.title}</button>)}</nav>
    </aside>
    <section className="flex min-h-[70vh] min-w-0 flex-col">
      <header className="flex items-center justify-between gap-3 border-b px-5 py-3"><div><h1 className="font-semibold">{chat.data?.title === "New chat" ? t("chat.newChat") : chat.data?.title ?? t("chat.title")}</h1><p className="text-xs text-muted-foreground">{dashboard.data?.inference_model ?? t("common.unavailable")}</p></div>{selected && chat.data ? <div className="flex gap-1"><Button variant="ghost" className="h-9 w-9 px-0" aria-label={t("chat.rename")} onClick={() => { const title = window.prompt(t("chat.rename"), chat.data.title); if (title?.trim()) rename.mutate({ id: selected, title }); }}><Pencil size={16} /></Button><Button variant="ghost" className="h-9 w-9 px-0" aria-label={t("common.delete")} onClick={() => { if (window.confirm(t("chat.deleteConfirm"))) remove.mutate(selected); }}><Trash2 size={16} /></Button></div> : null}</header>
      <div className="flex-1 overflow-auto px-4 py-6 sm:px-8">{!selected || !chat.data?.messages.length ? <div className="mx-auto flex max-w-xl flex-col items-center py-16 text-center"><div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-primary text-xl font-bold text-primary-foreground">A</div><h2 className="text-xl font-semibold">{t("chat.startConversation")}</h2><p className="mt-2 text-sm text-muted-foreground">{t("chat.localOnly")}</p><p className="mt-3 text-sm">{dashboard.data?.inference_model ?? t("common.unavailable")}</p></div> : <div className="mx-auto max-w-3xl space-y-6">{chat.data.messages.filter((message) => message.id !== visible?.messageId || !running).map((message) => <article key={message.id} className={message.role === "user" ? "ml-auto max-w-[85%] rounded-2xl bg-primary px-4 py-3 text-primary-foreground" : "max-w-full"}><div className="break-words text-sm"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</ReactMarkdown></div><button className="mt-2 opacity-60 hover:opacity-100" aria-label={t("chat.copy")} onClick={() => void navigator.clipboard.writeText(message.content)}><Copy size={14} /></button></article>)}{visible && running ? <article className="text-sm"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{visible.text}</ReactMarkdown><span className="animate-pulse">▍</span><p className="mt-2 text-xs text-muted-foreground">{t(visible.status === "prefill" || visible.status === "queued" ? "chat.preparing" : "chat.generating")}</p></article> : null}<div ref={bottom} /></div>}</div>
      {selected ? <form className="mx-auto w-full max-w-3xl p-4" onSubmit={submit}><div className="flex items-end gap-2 rounded-xl border bg-background p-2 shadow-sm"><textarea className="max-h-40 min-h-10 flex-1 resize-y bg-transparent p-2 text-sm outline-none" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder={t("chat.prompt")} aria-label={t("chat.prompt")} rows={1} />{running ? <Button type="button" aria-label={t("chat.stop")} onClick={() => send({ type: "generation.cancel", request_id: active.requestId })}><Square size={16} /></Button> : <Button type="submit" aria-label={t("chat.send")} disabled={!prompt.trim()}><Send size={16} /></Button>}</div>{error ? <p role="alert" className="pt-2 text-xs text-destructive">{t("chat.requestError")}</p> : null}</form> : null}
    </section>
  </div>;
}
