import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowDown, PanelLeft } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ChatSidebar } from "@/components/chat/chat-sidebar";
import { Composer } from "@/components/chat/composer";
import { MessageItem } from "@/components/chat/message-item";
import { SettingsDialog } from "@/components/chat/settings-dialog";
import { DEFAULT_SETTINGS } from "@/lib/chat-settings";
import { Button } from "@/components/ui/button";
import { useChatGeneration } from "@/hooks/use-chat-generation";
import { useModelRegistry, useRuntime } from "@/hooks/use-runtime";
import { api, type ChatMessage, type ChatSummary, type GenerationSettings, type Model } from "@/lib/api";
import { isActive } from "@/lib/generation";
import { defaultModel, modelById } from "@/lib/models";
import { useToastStore } from "@/store/toast-store";
import { useUiStore } from "@/store/ui-store";

const SUGGESTIONS = ["math", "code", "explain", "kazakh"] as const;
const STICK_THRESHOLD = 96;

function resolveSettings(saved: GenerationSettings | undefined): Required<GenerationSettings> {
  return { ...DEFAULT_SETTINGS, ...(saved ?? {}) } as Required<GenerationSettings>;
}

export function ChatPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const backendOnline = useUiStore((state) => state.backendOnline);
  const [params, setParams] = useSearchParams();
  const selected = params.get("c");
  const [query, setQuery] = useState("");
  const search = useDeferredValue(query);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [draftModel, setDraftModel] = useState<string | null>(null);
  const [pinned, setPinned] = useState(true);
  const viewport = useRef<HTMLDivElement>(null);
  const { active, failure, start, cancel, clearFailure } = useChatGeneration();

  const chats = useQuery({ queryKey: ["chats", search], queryFn: () => api.chats(search), staleTime: 30_000 });
  const chat = useQuery({ queryKey: ["chat", selected], queryFn: () => api.chat(selected ?? ""), enabled: selected !== null, staleTime: 30_000 });
  const registry = useModelRegistry().data;
  const runtime = useRuntime().data;

  const model: Model | null = useMemo(
    () => modelById(registry, chat.data?.checkpoint ?? draftModel) ?? modelById(registry, draftModel) ?? defaultModel(registry, runtime),
    [registry, runtime, chat.data?.checkpoint, draftModel],
  );
  const settings = useMemo(() => resolveSettings(chat.data?.generation_settings), [chat.data?.generation_settings]);
  const running = isActive(active);
  const visibleActive = active && active.chatId === selected ? active : null;

  const setSelected = useCallback((id: string | null) => setParams(id ? { c: id } : {}, { replace: true }), [setParams]);
  const refreshChats = () => void client.invalidateQueries({ queryKey: ["chats"] });
  const updateChat = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Parameters<typeof api.updateChat>[1] }) => api.updateChat(id, payload),
    onSuccess: (_, { id }) => {
      refreshChats();
      void client.invalidateQueries({ queryKey: ["chat", id] });
    },
    onError: (error: Error) => push("error", error.message),
  });
  const removeChat = useMutation({
    mutationFn: api.deleteChat,
    onSuccess: (_, id) => {
      if (id === selected) setSelected(null);
      refreshChats();
    },
  });
  const createChat = useMutation({ mutationFn: api.createChat });

  const messages: ChatMessage[] = useMemo(() => {
    const base = chat.data?.messages ?? [];
    if (!visibleActive) return base;
    let list = base;
    if (visibleActive.action === "edit" && visibleActive.targetId) {
      const index = list.findIndex((item) => item.id === visibleActive.targetId);
      if (index >= 0) list = [...list.slice(0, index), { ...list[index], content: visibleActive.pendingContent ?? list[index].content }];
    }
    if (visibleActive.action === "regenerate" && visibleActive.targetId) {
      const index = list.findIndex((item) => item.id === visibleActive.targetId);
      if (index >= 0) list = list.slice(0, index);
    }
    if (list.some((item) => item.id === visibleActive.messageId)) return list;
    const blank = { input_tokens: null, output_tokens: null, generation_ms: null, tokens_per_second: null, model_id: visibleActive.modelId, checkpoint_id: visibleActive.checkpointId, generation_params: null, reasoning_trace: [], created_at: "" };
    const extra: ChatMessage[] = [];
    if (visibleActive.action === "send" && visibleActive.pendingContent) extra.push({ ...blank, id: `pending-user-${visibleActive.requestId}`, role: "user", content: visibleActive.pendingContent, status: "completed" });
    if (visibleActive.action !== "continue") extra.push({ ...blank, id: visibleActive.messageId ?? `pending-${visibleActive.requestId}`, role: "assistant", content: "", status: "running" });
    return [...list, ...extra];
  }, [chat.data?.messages, visibleActive]);

  useEffect(() => {
    const element = viewport.current;
    if (element && pinned) element.scrollTop = element.scrollHeight;
  }, [messages, visibleActive?.text, visibleActive?.stages.length, pinned]);
  useEffect(() => setPinned(true), [selected]);

  const onScroll = () => {
    const element = viewport.current;
    if (element) setPinned(element.scrollHeight - element.scrollTop - element.clientHeight < STICK_THRESHOLD);
  };
  const modelPayload = (target: Model | null) => (target ? { model_name: target.model_id, checkpoint: target.id } : {});

  const send = async (content: string) => {
    if (!model) return;
    let chatId = selected;
    if (chatId === null) {
      const created = await createChat.mutateAsync({ ...modelPayload(model), generation_settings: settings });
      chatId = created.id;
      setSelected(chatId);
      refreshChats();
    }
    setPinned(true);
    start({ chatId, action: "send", content, checkpointId: model.id, settings });
  };
  const act = (action: "edit" | "regenerate" | "continue", message: ChatMessage, content?: string) => {
    if (selected && model) start({ chatId: selected, action, targetId: message.id, content, checkpointId: model.id, settings });
  };
  const pickModel = (next: Model) => {
    setDraftModel(next.id);
    if (selected) updateChat.mutate({ id: selected, payload: modelPayload(next) });
  };
  const lastAssistant = [...messages].reverse().find((item) => item.role === "assistant")?.id;
  const lastUsage = [...messages].reverse().find((item) => item.input_tokens !== null);
  const contextUsed = visibleActive?.inputTokens ?? (lastUsage ? (lastUsage.input_tokens ?? 0) + (lastUsage.output_tokens ?? 0) : null);
  const contextLimit = visibleActive?.contextLimit ?? model?.context_length ?? null;
  const stageNotice = model?.stage === "pretrain" ? t("chat.pretrainNotice") : null;
  const productionNotice = runtime?.production && !runtime.production.ready ? t("chat.notReadyNotice") : null;
  const failureHere = failure && failure.chatId === selected ? failure : null;
  const empty = messages.length === 0;
  const title = chat.data?.title ?? t("chat.newChat");

  return (
    <div className="flex h-full min-h-0">
      <aside className={`${sidebarOpen ? "fixed inset-y-0 left-0 z-40 flex shadow-xl" : "hidden"} w-72 shrink-0 flex-col border-r bg-card lg:static lg:flex lg:shadow-none`}>
        <ChatSidebar
          chats={chats.data ?? []}
          selected={selected}
          query={query}
          onQuery={setQuery}
          onSelect={(id) => { setSelected(id); setSidebarOpen(false); }}
          onNew={() => { setSelected(null); setSidebarOpen(false); }}
          onPin={(item: ChatSummary) => updateChat.mutate({ id: item.id, payload: { pinned: !item.pinned } })}
          onRename={(item, next) => updateChat.mutate({ id: item.id, payload: { title: next } })}
          onDelete={(item) => removeChat.mutate(item.id)}
        />
      </aside>
      {sidebarOpen ? <button type="button" aria-label={t("common.close")} className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setSidebarOpen(false)} /> : null}
      <section className="relative flex min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center gap-2 border-b px-3 lg:hidden">
          <Button variant="ghost" size="icon" className="h-8 w-8" aria-label={t("chat.history")} onClick={() => setSidebarOpen(true)}><PanelLeft size={16} /></Button>
          <span className="truncate text-sm font-medium">{title}</span>
        </div>
        <div ref={viewport} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto" data-testid="messages">
          <div className="mx-auto grid w-full max-w-3xl gap-6 px-4 py-6">
            {empty ? (
              <div className="grid gap-6 py-10 text-center sm:py-16">
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight">{t("chat.welcome")}</h1>
                  <p className="mt-1 text-sm text-muted-foreground">{t("chat.localOnly")}</p>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {SUGGESTIONS.map((key) => (
                    <button key={key} type="button" disabled={running || !model || !backendOnline} onClick={() => void send(t(`chat.suggestions.${key}`))} className="rounded-xl border bg-card p-3 text-left text-sm transition-colors hover:bg-muted disabled:opacity-50">
                      {t(`chat.suggestions.${key}`)}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <MessageItem
                  key={message.id}
                  message={message}
                  streaming={visibleActive && message.id === visibleActive.messageId ? visibleActive : visibleActive && message.role === "assistant" && message.id.startsWith("pending-") ? visibleActive : null}
                  isLast={message.id === lastAssistant}
                  busy={running}
                  onEdit={(item, content) => act("edit", item, content)}
                  onRegenerate={(item) => act("regenerate", item)}
                  onContinue={(item) => act("continue", item)}
                />
              ))
            )}
          </div>
        </div>
        {!pinned && !empty ? (
          <Button variant="outline" size="icon" className="absolute bottom-40 right-6 z-10 rounded-full shadow" aria-label={t("chat.scrollBottom")} onClick={() => { setPinned(true); }}><ArrowDown size={16} /></Button>
        ) : null}
        <div className="shrink-0 px-4 pb-3 pt-1">
          <div className="mx-auto grid max-w-3xl gap-2">
            {failureHere ? (
              <div role="alert" className="flex items-start justify-between gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <span>{t(`chatError.${failureHere.code}`, { defaultValue: failureHere.detail ?? t("chatError.generation_failed") })}</span>
                <button type="button" className="underline" onClick={clearFailure}>{t("common.close")}</button>
              </div>
            ) : null}
            {!backendOnline ? <p role="status" className="rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning">{t("chatError.disconnected")}</p> : null}
            {stageNotice || productionNotice ? (
              <p role="note" className="flex items-start gap-2 rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning"><AlertTriangle size={13} className="mt-0.5 shrink-0" />{stageNotice ?? productionNotice}</p>
            ) : null}
            <Composer
              registry={registry}
              model={model}
              generating={running}
              offline={!backendOnline}
              contextUsed={contextUsed}
              contextLimit={contextLimit}
              onSend={(content) => void send(content)}
              onStop={cancel}
              onModel={pickModel}
              onSettings={() => setSettingsOpen(true)}
              focusKey={selected}
            />
          </div>
        </div>
      </section>
      <SettingsDialog
        key={`${selected}-${settingsOpen}`}
        open={settingsOpen}
        settings={settings}
        systemPrompt={chat.data?.system_prompt ?? ""}
        onClose={() => setSettingsOpen(false)}
        onSave={(next, prompt) => selected ? updateChat.mutate({ id: selected, payload: { generation_settings: next, system_prompt: prompt } }) : push("info", t("chat.settingsAfterFirst"))}
      />
    </div>
  );
}
