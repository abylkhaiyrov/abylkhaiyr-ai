import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowDown, PanelLeft, SlidersHorizontal } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ChatSidebar } from "@/components/chat/chat-sidebar";
import { Composer } from "@/components/chat/composer";
import { ContextPanel } from "@/components/chat/context-panel";
import { MessageItem } from "@/components/chat/message-item";
import { ModelPicker } from "@/components/chat/model-picker";
import { SettingsDialog } from "@/components/chat/settings-dialog";
import { Button } from "@/components/ui/button";
import { ErrorState, LoadingState } from "@/components/data-state";
import { useApiError } from "@/hooks/use-api-error";
import { useChatGeneration } from "@/hooks/use-chat-generation";
import { useRole } from "@/hooks/use-auth";
import { useModelRegistry, useRuntime } from "@/hooks/use-runtime";
import { api, type ChatMessage, type ChatSummary, type GenerationSettings, type Model } from "@/lib/api";
import { DEFAULT_SETTINGS } from "@/lib/chat-settings";
import { isActive } from "@/lib/generation";
import { defaultModel, modelById } from "@/lib/models";
import { cn } from "@/lib/utils";
import { useToastStore } from "@/store/toast-store";
import { useUiStore } from "@/store/ui-store";

const SUGGESTIONS = ["math", "code", "explain", "kazakh"] as const;
const STICK_THRESHOLD = 96;
const MESSAGE_PAGE = 100;

const resolveSettings = (saved: GenerationSettings | undefined): Required<GenerationSettings> => ({ ...DEFAULT_SETTINGS, ...(saved ?? {}) }) as Required<GenerationSettings>;

export function ChatPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const backendOnline = useUiStore((state) => state.socketStatus === "connected");
  const { historyOpen, toggleHistory } = useUiStore();
  const [params, setParams] = useSearchParams();
  const selected = params.get("c");
  const [query, setQuery] = useState("");
  const search = useDeferredValue(query);
  const [drawer, setDrawer] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [draftModel, setDraftModel] = useState<string | null>(null);
  const [pinned, setPinned] = useState(true);
  const viewport = useRef<HTMLDivElement>(null);
  const { active, failure, start, cancel, clearFailure } = useChatGeneration();

  const isAdmin = useRole() === "ADMIN";
  const [archived, setArchived] = useState(false);
  const [messageLimit, setMessageLimit] = useState(MESSAGE_PAGE);
  const chats = useQuery({ queryKey: ["chats", search, archived], queryFn: () => api.chats(search, archived), staleTime: 30_000 });
  const chat = useQuery({ queryKey: ["chat", selected, messageLimit], queryFn: () => api.chat(selected ?? "", messageLimit), enabled: selected !== null, staleTime: 30_000, placeholderData: (previous) => previous });
  const activeModelInfo = useQuery({ queryKey: ["chat-model"], queryFn: api.chatModel, enabled: !isAdmin, staleTime: 60_000 });
  const registry = useModelRegistry().data;
  const runtime = useRuntime().data;

  const adminModel: Model | null = useMemo(
    () => modelById(registry, chat.data?.checkpoint ?? draftModel) ?? modelById(registry, draftModel) ?? defaultModel(registry, runtime),
    [registry, runtime, chat.data?.checkpoint, draftModel],
  );
  const model: Pick<Model, "id" | "display_name" | "stage" | "context_length"> | null = isAdmin ? adminModel : (activeModelInfo.data ?? null);
  const settings = useMemo(() => resolveSettings(chat.data?.generation_settings), [chat.data?.generation_settings]);
  const running = isActive(active);
  const visibleActive = active && active.chatId === selected ? active : null;

  const setSelected = useCallback((id: string | null) => setParams(id ? { c: id } : {}, { replace: true }), [setParams]);
  const refreshChats = () => void client.invalidateQueries({ queryKey: ["chats"] });
  const refreshChat = (id: string) => void client.invalidateQueries({ queryKey: ["chat", id] });
  const updateChat = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Parameters<typeof api.updateChat>[1] }) => api.updateChat(id, payload),
    onSuccess: (_, { id }) => {
      refreshChats();
      refreshChat(id);
    },
    onError: (error: Error) => push("error", describe(error)),
  });
  const removeChat = useMutation({
    mutationFn: api.deleteChat,
    onSuccess: (_, id) => {
      if (id === selected) setSelected(null);
      refreshChats();
    },
    onError: (error: Error) => push("error", describe(error)),
  });
  const useVariant = useMutation({
    mutationFn: ({ message, index }: { message: ChatMessage; index: number }) => api.selectVariant(selected ?? "", message.id, index),
    onSuccess: () => selected && refreshChat(selected),
    onError: (error: Error) => push("error", describe(error)),
  });
  const createChat = useMutation({ mutationFn: api.createChat });

  const messages: ChatMessage[] = useMemo(() => {
    const base = chat.data?.messages ?? [];
    if (!visibleActive) return base;
    let list = base;
    const targetIndex = visibleActive.targetId ? list.findIndex((item) => item.id === visibleActive.targetId) : -1;
    if (visibleActive.action === "edit" && targetIndex >= 0) list = [...list.slice(0, targetIndex), { ...list[targetIndex], content: visibleActive.pendingContent ?? list[targetIndex].content }];
    if (visibleActive.action === "regenerate" && targetIndex >= 0) list = list.slice(0, targetIndex);
    if (list.some((item) => item.id === visibleActive.messageId)) return list;
    const blank = { input_tokens: null, output_tokens: null, generation_ms: null, tokens_per_second: null, model_id: visibleActive.modelId, checkpoint_id: visibleActive.checkpointId, generation_params: null, reasoning_trace: [], variants: [], created_at: "" };
    const extra: ChatMessage[] = [];
    if (visibleActive.action === "send" && visibleActive.pendingContent) extra.push({ ...blank, id: `pending-user-${visibleActive.requestId}`, role: "user", content: visibleActive.pendingContent, status: "completed" });
    if (visibleActive.action !== "continue") extra.push({ ...blank, id: visibleActive.messageId ?? `pending-${visibleActive.requestId}`, role: "assistant", content: "", status: "running" });
    return [...list, ...extra];
  }, [chat.data?.messages, visibleActive]);

  useEffect(() => {
    const element = viewport.current;
    if (element && pinned) element.scrollTop = element.scrollHeight;
  }, [messages, visibleActive?.text, visibleActive?.stages.length, pinned]);
  useEffect(() => {
    setPinned(true);
    setMessageLimit(MESSAGE_PAGE);
  }, [selected]);

  const onScroll = () => {
    const element = viewport.current;
    if (element) setPinned(element.scrollHeight - element.scrollTop - element.clientHeight < STICK_THRESHOLD);
  };
  const modelPayload = (target: Model | null) => (isAdmin && target ? { model_name: target.model_id, checkpoint: target.id } : {});

  const send = async (content: string) => {
    if (!model || running || createChat.isPending) return;
    let chatId = selected;
    if (chatId === null) {
      try {
        const created = await createChat.mutateAsync({ ...modelPayload(adminModel), generation_settings: settings });
        chatId = created.id;
        setSelected(chatId);
        refreshChats();
      } catch (error) {
        push("error", describe(error));
        return;
      }
    }
    setPinned(true);
    start({ chatId, action: "send", content, checkpointId: isAdmin ? model.id : null, settings });
  };
  const act = (action: "edit" | "regenerate" | "continue", message: ChatMessage, content?: string) => {
    if (selected && model) start({ chatId: selected, action, targetId: message.id, content, checkpointId: isAdmin ? model.id : null, settings });
  };
  const pickModel = (next: Model) => {
    setDraftModel(next.id);
    if (selected) updateChat.mutate({ id: selected, payload: modelPayload(next) });
  };
  const saveSettings = (next: Required<GenerationSettings>, prompt: string) => {
    if (selected) updateChat.mutate({ id: selected, payload: { generation_settings: next, system_prompt: prompt } });
    else push("info", t("chat.settingsAfterFirst"));
  };

  const lastAssistant = [...messages].reverse().find((item) => item.role === "assistant")?.id;
  const lastUsage = [...messages].reverse().find((item) => item.input_tokens !== null);
  const contextUsed = visibleActive?.inputTokens ?? (lastUsage ? (lastUsage.input_tokens ?? 0) + (lastUsage.output_tokens ?? 0) : null);
  const contextLimit = visibleActive?.contextLimit ?? model?.context_length ?? null;
  const notice = model?.stage === "pretrain" ? t("chat.pretrainNotice") : runtime?.production && !runtime.production.ready ? t("chat.notReadyNotice") : null;
  const failureHere = failure && failure.chatId === selected ? failure : null;
  const empty = messages.length === 0;
  const blockReason = !backendOnline ? t("chatError.disconnected") : model === null ? t("chat.noModel") : null;
  const composer = (
    <Composer generating={running} disabled={blockReason !== null || createChat.isPending} disabledReason={blockReason} onSend={(content) => void send(content)} onStop={cancel} focusKey={selected} large={empty} />
  );
  const sidebar = (
    <ChatSidebar
      chats={chats.data ?? []}
      selected={selected}
      query={query}
      onQuery={setQuery}
      onSelect={(id) => { setSelected(id); setDrawer(false); }}
      onNew={() => { setSelected(null); setDrawer(false); }}
      archived={archived}
      onArchivedChange={setArchived}
      onArchive={(item: ChatSummary) => updateChat.mutate({ id: item.id, payload: { archived: !item.archived } })}
      onPin={(item: ChatSummary) => updateChat.mutate({ id: item.id, payload: { pinned: !item.pinned } })}
      onRename={(item, next) => updateChat.mutate({ id: item.id, payload: { title: next } })}
      onDelete={(item) => removeChat.mutate(item.id)}
    />
  );

  return (
    <div className="flex h-full min-h-0">
      <aside aria-label={t("chat.history")} className={cn("hidden w-64 shrink-0 border-r hairline", historyOpen && "lg:block")}>{sidebar}</aside>
      {drawer ? (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button type="button" aria-label={t("common.close")} className="absolute inset-0 bg-black/40" onClick={() => setDrawer(false)} />
          <aside aria-label={t("chat.history")} className="relative h-full w-72 max-w-[85vw] bg-background shadow-xl">{sidebar}</aside>
        </div>
      ) : null}
      <section className="relative flex min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center gap-1 px-2">
          <Button variant="ghost" size="icon" className="lg:hidden" aria-label={t("chat.history")} onClick={() => setDrawer(true)}><PanelLeft size={16} /></Button>
          <Button variant="ghost" size="icon" className="hidden lg:inline-flex" aria-label={t(historyOpen ? "chat.hideHistory" : "chat.showHistory")} aria-pressed={historyOpen} onClick={toggleHistory}><PanelLeft size={16} /></Button>
          {isAdmin ? (
            <ModelPicker registry={registry} selected={adminModel} locked={running} onSelect={pickModel} />
          ) : (
            <span className="px-2 text-[13px] font-semibold" title={t("chat.activeModel")}>{model?.display_name ?? t("chat.noModel")}</span>
          )}
          <span className="mx-1 hidden min-w-0 flex-1 truncate text-center text-[13px] text-muted-foreground md:block">{chat.data?.title ?? ""}</span>
          <div className="ml-auto flex items-center md:ml-0">
            <ContextPanel
              used={contextUsed}
              limit={contextLimit}
              messageCount={messages.length}
              window={settings.context_messages}
              disabled={selected === null || running}
              onWindow={(value) => selected && updateChat.mutate({ id: selected, payload: { generation_settings: { ...settings, context_messages: value } } })}
              onSystemPrompt={() => setSettingsOpen(true)}
              onClear={() => setSelected(null)}
            />
            <Button variant="ghost" size="icon" aria-label={t("chat.settings")} title={t("chat.settings")} onClick={() => setSettingsOpen(true)}><SlidersHorizontal size={15} /></Button>
          </div>
        </div>
        <div ref={viewport} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto" data-testid="messages">
          {selected !== null && chat.isPending ? (
            <div className="mx-auto w-full max-w-[44rem] p-4"><LoadingState /></div>
          ) : selected !== null && chat.isLoadingError ? (
            <div className="mx-auto w-full max-w-[44rem] p-4"><ErrorState error={chat.error} retry={() => void chat.refetch()} /></div>
          ) : empty ? (
            <div className="mx-auto flex min-h-full w-full max-w-[44rem] flex-col justify-center gap-6 px-4 pb-10">
              <div className="text-center">
                <h1 className="text-[28px] font-semibold tracking-tight">{t("chat.welcome")}</h1>
                <p className="mt-1.5 text-[13px] text-muted-foreground">{t("chat.localOnly")}</p>
              </div>
              <div className="grid gap-2">
                {notice ? <p role="note" className="flex items-start gap-2 text-xs text-warning"><AlertTriangle size={13} className="mt-0.5 shrink-0" />{notice}</p> : null}
                {failureHere ? <FailureNotice code={failureHere.code} detail={failureHere.detail} onClose={clearFailure} /> : null}
                {composer}
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((key) => (
                  <button key={key} type="button" disabled={running || createChat.isPending || blockReason !== null} onClick={() => void send(t(`chat.suggestions.${key}`))} className="rounded-full border px-3 py-1.5 text-[13px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50">
                    {t(`chat.suggestions.${key}`)}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto grid w-full max-w-[44rem] gap-7 px-4 pb-6 pt-4">
              {chat.data && chat.data.total_messages > chat.data.messages.length ? (
                <Button variant="ghost" size="sm" className="mx-auto" disabled={chat.isFetching} onClick={() => setMessageLimit((value) => value + MESSAGE_PAGE)}>{t("chat.loadEarlier")}</Button>
              ) : null}
              {messages.map((message) => (
                <MessageItem
                  key={message.id}
                  message={message}
                  streaming={visibleActive && (message.id === visibleActive.messageId || (message.role === "assistant" && message.id.startsWith("pending-"))) ? visibleActive : null}
                  isLast={message.id === lastAssistant}
                  busy={running}
                  onEdit={(item, content) => act("edit", item, content)}
                  onRegenerate={(item) => act("regenerate", item)}
                  onContinue={(item) => act("continue", item)}
                  onUseVariant={(item, index) => useVariant.mutate({ message: item, index })}
                />
              ))}
            </div>
          )}
        </div>
        {!pinned && !empty ? (
          <Button variant="outline" size="icon" className="absolute bottom-36 right-6 z-10 rounded-full bg-background shadow" aria-label={t("chat.scrollBottom")} onClick={() => setPinned(true)}><ArrowDown size={15} /></Button>
        ) : null}
        {empty ? null : (
          <div className="shrink-0 px-4 pb-3 pt-1">
            <div className="mx-auto grid max-w-[44rem] gap-2">
              {failureHere ? <FailureNotice code={failureHere.code} detail={failureHere.detail} onClose={clearFailure} /> : null}
              {notice ? <p role="note" className="flex items-start gap-2 text-xs text-warning"><AlertTriangle size={13} className="mt-0.5 shrink-0" />{notice}</p> : null}
              {composer}
            </div>
          </div>
        )}
      </section>
      <SettingsDialog key={`${selected}-${settingsOpen}`} open={settingsOpen} settings={settings} systemPrompt={chat.data?.system_prompt ?? ""} onClose={() => setSettingsOpen(false)} onSave={saveSettings} />
    </div>
  );
}

function FailureNotice({ code, detail, onClose }: { code: string; detail: string | null; onClose: () => void }) {
  const { t, i18n } = useTranslation();
  const key = `chatError.${code}`;
  return (
    <div role="alert" className="flex items-start justify-between gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
      <span>{i18n.exists(key) ? t(key) : detail ? t("chatError.generation_failed") : t("chatError.generation_failed")}</span>
      <button type="button" className="underline" onClick={onClose}>{t("common.close")}</button>
    </div>
  );
}
