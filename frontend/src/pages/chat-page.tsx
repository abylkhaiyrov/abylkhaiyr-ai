import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  ArrowDown,
  Check,
  Copy,
  Cpu,
  FastForward,
  Loader2,
  PanelLeft,
  Pencil,
  Plus,
  Power,
  RefreshCw,
  Send,
  SlidersHorizontal,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { api, type GenerationSettings, type Model, type ReasoningStageEvent } from "@/lib/api";
import { useStudioEvents, type StudioEvent } from "@/hooks/use-studio-events";
import { useUiStore } from "@/store/ui-store";
import { Button } from "@/components/ui/button";
import { ErrorState, LoadingState } from "@/components/data-state";
import { MarkdownContent } from "@/components/markdown-content";
import { ReasoningPanel } from "@/components/reasoning-panel";

type GenerationMode = "standard" | "reasoning" | "precise" | "creative";

type ActiveGeneration = {
  requestId: string;
  chatId: string;
  messageId?: string;
  text: string;
  sequence: number;
  status: string;
  queuePosition?: number;
  queueReason?: string | null;
  modelId?: string | null;
  checkpointId?: string | null;
  device?: string | null;
  inputTokens?: number;
  contextLimit?: number;
  reasoningEvents: ReasoningStageEvent[];
};

const DEFAULT_SETTINGS: Required<GenerationSettings> = {
  mode: "standard",
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  max_tokens: 128,
  repetition_penalty: 1.1,
};

const MODES: GenerationMode[] = ["standard", "reasoning", "precise", "creative"];

export function ChatPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const { send, subscribe } = useStudioEvents();
  const backendOnline = useUiStore((state) => state.backendOnline);
  const [selected, setSelected] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [search, setSearch] = useState("");
  const [active, setActive] = useState<ActiveGeneration | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [settings, setSettings] = useState<Required<GenerationSettings>>(DEFAULT_SETTINGS);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [pinnedToBottom, setPinnedToBottom] = useState(true);
  const viewportRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const chats = useQuery({ queryKey: ["chats", search], queryFn: () => api.chats(search) });
  const dashboard = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const chat = useQuery({
    queryKey: ["chat", selected],
    queryFn: () => api.chat(selected ?? ""),
    enabled: selected !== null,
  });

  const activeModel = useMemo<Model | null>(() => {
    const loaded = models.data?.find((item) => item.loaded);
    return loaded ?? dashboard.data?.active_model ?? null;
  }, [models.data, dashboard.data?.active_model]);

  const create = useMutation({
    mutationFn: () =>
      api.createChat({
        model_name: activeModel?.model_id,
        checkpoint: activeModel?.checkpoint_id,
        generation_settings: settings,
      }),
    onSuccess: (item) => {
      void client.invalidateQueries({ queryKey: ["chats"] });
      setSelected(item.id);
    },
  });

  const remove = useMutation({
    mutationFn: api.deleteChat,
    onSuccess: () => {
      setSelected(null);
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });

  const updateChat = useMutation({
    mutationFn: ({
      id,
      payload,
    }: {
      id: string;
      payload: {
        title?: string;
        model_name?: string;
        checkpoint?: string;
        system_prompt?: string;
        generation_settings?: GenerationSettings;
      };
    }) => api.updateChat(id, payload),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["chats"] });
      void client.invalidateQueries({ queryKey: ["chat", selected] });
    },
  });

  const switchModel = useMutation({
    mutationFn: (checkpointId: string) => api.switchModel(checkpointId),
    onSuccess: (loaded) => {
      setErrorCode(null);
      setErrorDetail(null);
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["dashboard"] });
      if (selected) {
        updateChat.mutate({
          id: selected,
          payload: { model_name: loaded.model_id, checkpoint: loaded.checkpoint_id },
        });
      }
    },
    onError: (err: Error) => {
      setErrorCode("generation_failed");
      setErrorDetail(err.message);
    },
  });

  const unloadModel = useMutation({
    mutationFn: api.unloadModel,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });

  useEffect(() => {
    if (!chat.data) return;
    const saved = chat.data.generation_settings ?? {};
    setSettings({
      mode: (saved.mode as GenerationMode) ?? DEFAULT_SETTINGS.mode,
      temperature: typeof saved.temperature === "number" ? saved.temperature : DEFAULT_SETTINGS.temperature,
      top_p: typeof saved.top_p === "number" ? saved.top_p : DEFAULT_SETTINGS.top_p,
      top_k: typeof saved.top_k === "number" ? saved.top_k : DEFAULT_SETTINGS.top_k,
      max_tokens: typeof saved.max_tokens === "number" ? saved.max_tokens : DEFAULT_SETTINGS.max_tokens,
      repetition_penalty:
        typeof saved.repetition_penalty === "number"
          ? saved.repetition_penalty
          : DEFAULT_SETTINGS.repetition_penalty,
    });
    setSystemPrompt(chat.data.system_prompt ?? "");
  }, [chat.data]);

  useEffect(
    () =>
      subscribe((event: StudioEvent) => {
        if (!event.type.startsWith("generation.") || !event.request_id) return;
        if (event.type === "generation.started") {
          void client.invalidateQueries({ queryKey: ["chat"] });
          void client.invalidateQueries({ queryKey: ["chats"] });
        }
        if (event.type === "generation.failed") {
          setErrorCode(event.code ?? "generation_failed");
          setErrorDetail(event.detail ?? null);
        }
        if (
          ["generation.completed", "generation.cancelled", "generation.failed"].includes(event.type)
        ) {
          void client.invalidateQueries({ queryKey: ["chat"] });
          void client.invalidateQueries({ queryKey: ["chats"] });
          void client.invalidateQueries({ queryKey: ["models"] });
          void client.invalidateQueries({ queryKey: ["dashboard"] });
        }
        setActive((previous) => {
          if (event.type === "generation.snapshot" && event.chat_id) {
            if (
              previous &&
              previous.requestId === event.request_id &&
              (event.sequence_id ?? 0) < previous.sequence
            ) {
              return previous;
            }
            return {
              requestId: event.request_id!,
              chatId: event.chat_id,
              messageId: event.message_id,
              text: event.text ?? "",
              sequence: event.sequence_id ?? 0,
              status: event.status ?? "running",
              queuePosition: event.queue_position,
              queueReason: event.queue_reason,
              modelId: event.model_id,
              checkpointId: event.checkpoint_id,
              device: event.device,
              reasoningEvents: event.reasoning_events ?? [],
            };
          }
          if (
            !previous ||
            previous.requestId !== event.request_id ||
            (event.sequence_id ?? 0) <= previous.sequence
          ) {
            return previous;
          }
          const nextStatus =
            event.type === "generation.stage"
              ? previous.status
              : event.type.slice("generation.".length);
          const nextText =
            event.type === "generation.delta"
              ? previous.text + (event.text ?? "")
              : typeof event.text === "string"
                ? event.text
                : previous.text;
          return {
            ...previous,
            messageId: event.message_id ?? previous.messageId,
            sequence: event.sequence_id ?? previous.sequence,
            text: nextText,
            status: nextStatus,
            queuePosition: event.queue_position ?? previous.queuePosition,
            queueReason: event.queue_reason !== undefined ? event.queue_reason : previous.queueReason,
            modelId: event.model_id ?? previous.modelId,
            checkpointId: event.checkpoint_id ?? previous.checkpointId,
            device: event.device ?? previous.device,
            inputTokens: event.input_tokens ?? previous.inputTokens,
            contextLimit: event.context_limit ?? previous.contextLimit,
            reasoningEvents: event.reasoning_events ?? previous.reasoningEvents,
          };
        });
      }),
    [subscribe, client],
  );

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    bottomRef.current?.scrollIntoView({ behavior });
    setPinnedToBottom(true);
  }, []);

  useEffect(() => {
    if (pinnedToBottom) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [active?.text, active?.reasoningEvents.length, chat.data?.messages.length, pinnedToBottom]);

  function handleViewportScroll() {
    const element = viewportRef.current;
    if (!element) return;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    setPinnedToBottom(distance < 96);
  }

  const running =
    active !== null && ["queued", "started", "prefill", "delta", "running"].includes(active.status);
  const visibleActive = active?.chatId === selected ? active : null;

  const contextLimit =
    visibleActive?.contextLimit ?? activeModel?.context_length ?? 512;
  const latestMessageWithTokens = useMemo(() => {
    const list = chat.data?.messages ?? [];
    for (let i = list.length - 1; i >= 0; i -= 1) {
      if (list[i].input_tokens !== null || list[i].output_tokens !== null) {
        return list[i];
      }
    }
    return null;
  }, [chat.data?.messages]);
  const estimatedUsedTokens = useMemo(() => {
    if (visibleActive?.inputTokens) {
      return visibleActive.inputTokens + Math.ceil(visibleActive.text.length / 4);
    }
    if (latestMessageWithTokens) {
      return (
        (latestMessageWithTokens.input_tokens ?? 0) +
        (latestMessageWithTokens.output_tokens ?? 0)
      );
    }
    const totalChars =
      (chat.data?.messages ?? []).reduce((sum, item) => sum + item.content.length, 0) +
      systemPrompt.length;
    return Math.ceil(totalChars / 4);
  }, [visibleActive, latestMessageWithTokens, chat.data?.messages, systemPrompt.length]);

  function dispatchGeneration(options: {
    action: "send" | "edit" | "regenerate" | "continue";
    content?: string;
    targetMessageId?: string;
  }) {
    if (!selected || running) return;
    const requestId = crypto.randomUUID();
    const payload: Record<string, unknown> = {
      type: "generation.start",
      request_id: requestId,
      chat_id: selected,
      action: options.action,
      mode: settings.mode,
      temperature: settings.temperature,
      top_p: settings.top_p,
      top_k: settings.top_k,
      max_tokens: settings.max_tokens,
      repetition_penalty: settings.repetition_penalty,
    };
    if (options.content !== undefined) payload.content = options.content;
    if (options.targetMessageId !== undefined) payload.target_message_id = options.targetMessageId;
    if (activeModel?.checkpoint_id) payload.checkpoint_id = activeModel.checkpoint_id;
    if (!send(payload)) {
      setErrorCode("disconnected");
      setErrorDetail(null);
      return;
    }
    setActive({
      requestId,
      chatId: selected,
      messageId: options.action === "continue" ? options.targetMessageId : undefined,
      text: "",
      sequence: 0,
      status: "queued",
      modelId: activeModel?.model_id ?? null,
      checkpointId: activeModel?.checkpoint_id ?? null,
      device: activeModel?.device ?? null,
      reasoningEvents: [],
    });
    setErrorCode(null);
    setErrorDetail(null);
    setPinnedToBottom(true);
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected || !prompt.trim() || running) return;
    const content = prompt.trim();
    setPrompt("");
    dispatchGeneration({ action: "send", content });
  }

  function copyText(id: string, content: string) {
    void navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId((prev) => (prev === id ? null : prev)), 1500);
  }

  function saveSettings() {
    if (!selected) return;
    updateChat.mutate({
      id: selected,
      payload: {
        system_prompt: systemPrompt,
        generation_settings: settings,
      },
    });
  }

  function resolveErrorMessage(): string {
    if (!errorCode) return "";
    const translated = t(`chat.errors.${errorCode}`, { defaultValue: "" });
    if (translated) return errorDetail ? `${translated} (${errorDetail})` : translated;
    return errorDetail || t("chat.requestError");
  }

  if (chats.isPending) return <LoadingState />;
  if (chats.isError) return <ErrorState error={chats.error} retry={() => chats.refetch()} />;

  const availableModels = models.data ?? [];
  const selectedCheckpointId =
    activeModel?.checkpoint_id ?? chat.data?.checkpoint ?? availableModels[0]?.checkpoint_id ?? "";
  const contextPercent = Math.min(100, Math.round((estimatedUsedTokens / Math.max(1, contextLimit)) * 100));

  return (
    <div className="grid min-h-[calc(100vh-8rem)] overflow-hidden rounded-xl border bg-card shadow-sm md:grid-cols-[auto_minmax(0,1fr)]">
      {showHistory ? (
        <aside className="flex w-full flex-col border-b bg-muted/20 p-3 md:w-64 md:border-b-0 md:border-r">
          <Button
            className="w-full gap-2"
            onClick={() => create.mutate()}
            disabled={create.isPending}
          >
            <Plus size={16} />
            {t("chat.newChat")}
          </Button>
          <input
            className="my-3 w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={t("chat.searchChats")}
            aria-label={t("chat.searchChats")}
          />
          <nav
            className="grid max-h-56 gap-1 overflow-y-auto md:max-h-[calc(100vh-16rem)]"
            aria-label={t("chat.searchChats")}
          >
            {chats.data.length === 0 ? (
              <p className="px-2 py-4 text-center text-xs text-muted-foreground">
                {t("chat.noChats")}
              </p>
            ) : (
              chats.data.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`flex flex-col rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                    selected === item.id
                      ? "bg-primary/10 font-medium text-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                  onClick={() => {
                    setSelected(item.id);
                    setEditingMessageId(null);
                  }}
                >
                  <span className="truncate">
                    {item.title === "New chat" ? t("chat.newChat") : item.title}
                  </span>
                  {item.model_name ? (
                    <span className="truncate text-[11px] text-muted-foreground">
                      {item.model_name}
                    </span>
                  ) : null}
                </button>
              ))
            )}
          </nav>
        </aside>
      ) : null}

      <section className="relative flex min-h-[70vh] min-w-0 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b bg-card/95 px-4 py-3 backdrop-blur">
          <div className="flex min-w-0 items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              className="h-9 w-9 px-0"
              aria-label={t("chat.toggleHistory")}
              onClick={() => setShowHistory((prev) => !prev)}
            >
              <PanelLeft size={16} />
            </Button>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold sm:text-base">
                {chat.data?.title === "New chat"
                  ? t("chat.newChat")
                  : chat.data?.title ?? t("chat.title")}
              </h1>
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <Cpu size={12} />
                  {activeModel ? (
                    <>
                      <span className="font-medium text-foreground">{activeModel.model_id}</span>
                      <span className="rounded bg-muted px-1.5 py-0.5 uppercase">
                        {t(`resources.models.stages.${activeModel.stage}`, {
                          defaultValue: activeModel.stage,
                        })}
                      </span>
                      {activeModel.device ? (
                        <span className="rounded bg-muted px-1.5 py-0.5 font-mono uppercase">
                          {activeModel.device}
                        </span>
                      ) : null}
                    </>
                  ) : (
                    <span>{t("chat.noModelLoaded")}</span>
                  )}
                </span>
                <span
                  className="inline-flex items-center gap-1.5 rounded-full border bg-muted/40 px-2 py-0.5"
                  title={t("chat.contextLength")}
                >
                  <span>{t("chat.contextLength")}:</span>
                  <span className="font-mono text-foreground">
                    {t("chat.contextUsage", {
                      used: estimatedUsedTokens,
                      limit: contextLimit,
                    })}
                  </span>
                  <span className="h-1.5 w-10 overflow-hidden rounded-full bg-muted">
                    <span
                      className={`block h-full rounded-full ${
                        contextPercent > 85 ? "bg-destructive" : "bg-primary"
                      }`}
                      style={{ width: `${contextPercent}%` }}
                    />
                  </span>
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            <label className="sr-only" htmlFor="chat-checkpoint-select">
              {t("chat.selectModel")}
            </label>
            <select
              id="chat-checkpoint-select"
              aria-label={t("chat.selectModel")}
              className="h-9 max-w-[14rem] rounded-lg border bg-background px-2.5 text-xs outline-none focus:ring-1 focus:ring-primary"
              value={selectedCheckpointId}
              disabled={switchModel.isPending || running}
              onChange={(event) => {
                if (event.target.value) {
                  switchModel.mutate(event.target.value);
                }
              }}
            >
              {availableModels.length === 0 ? (
                <option value="">{t("chat.noModelLoaded")}</option>
              ) : (
                availableModels.map((item) => (
                  <option
                    key={item.checkpoint_id}
                    value={item.checkpoint_id}
                    disabled={!item.compatible}
                  >
                    {item.model_id}
                    {item.step ? ` · #${item.step}` : ""} ({item.stage.toUpperCase()})
                    {item.loaded ? " ✓" : ""}
                    {!item.compatible ? ` · ${t("resources.models.incompatible")}` : ""}
                  </option>
                ))
              )}
            </select>
            {activeModel ? (
              <Button
                type="button"
                variant="outline"
                className="h-9 gap-1 px-2.5 text-xs"
                disabled={unloadModel.isPending || running}
                onClick={() => unloadModel.mutate()}
                title={t("chat.unloadModel")}
              >
                <Power size={13} />
                <span className="hidden sm:inline">{t("chat.unloadModel")}</span>
              </Button>
            ) : selectedCheckpointId ? (
              <Button
                type="button"
                variant="outline"
                className="h-9 gap-1 px-2.5 text-xs"
                disabled={switchModel.isPending || running}
                onClick={() => switchModel.mutate(selectedCheckpointId)}
              >
                <Power size={13} />
                <span>{t("chat.loadModel")}</span>
              </Button>
            ) : null}
            <Button
              type="button"
              variant={showSettings ? "outline" : "ghost"}
              className="h-9 w-9 px-0"
              aria-label={t("chat.settings")}
              aria-expanded={showSettings}
              onClick={() => setShowSettings((prev) => !prev)}
            >
              <SlidersHorizontal size={16} />
            </Button>
            {selected && chat.data ? (
              <>
                <Button
                  type="button"
                  variant="ghost"
                  className="h-9 w-9 px-0"
                  aria-label={t("chat.rename")}
                  onClick={() => {
                    const title = window.prompt(t("chat.rename"), chat.data.title);
                    if (title?.trim()) {
                      updateChat.mutate({ id: selected, payload: { title } });
                    }
                  }}
                >
                  <Pencil size={15} />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className="h-9 w-9 px-0"
                  aria-label={t("common.delete")}
                  onClick={() => {
                    if (window.confirm(t("chat.deleteConfirm"))) remove.mutate(selected);
                  }}
                >
                  <Trash2 size={15} />
                </Button>
              </>
            ) : null}
          </div>
        </header>

        {switchModel.isPending ? (
          <div className="flex items-center gap-2 border-b bg-primary/5 px-4 py-2 text-xs text-primary">
            <Loader2 size={13} className="animate-spin" />
            <span>{t("chat.switchingModel")}</span>
          </div>
        ) : null}

        {showSettings ? (
          <div className="border-b bg-muted/30 px-4 py-4">
            <div className="mx-auto grid max-w-4xl gap-4 sm:grid-cols-2 lg:grid-cols-5">
              <label className="grid gap-1 text-xs">
                <span className="flex justify-between">
                  <span>{t("chat.temperature")}</span>
                  <span className="font-mono">{settings.temperature.toFixed(2)}</span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={settings.temperature}
                  onChange={(event) =>
                    setSettings((prev) => ({
                      ...prev,
                      temperature: Number(event.target.value),
                    }))
                  }
                />
              </label>
              <label className="grid gap-1 text-xs">
                <span className="flex justify-between">
                  <span>{t("chat.topP")}</span>
                  <span className="font-mono">{settings.top_p.toFixed(2)}</span>
                </span>
                <input
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={settings.top_p}
                  onChange={(event) =>
                    setSettings((prev) => ({ ...prev, top_p: Number(event.target.value) }))
                  }
                />
              </label>
              <label className="grid gap-1 text-xs">
                <span className="flex justify-between">
                  <span>{t("chat.topK")}</span>
                  <span className="font-mono">{settings.top_k}</span>
                </span>
                <input
                  type="number"
                  min={0}
                  max={500}
                  className="rounded-md border bg-background px-2 py-1 text-xs"
                  value={settings.top_k}
                  onChange={(event) =>
                    setSettings((prev) => ({
                      ...prev,
                      top_k: Math.max(0, Number(event.target.value) || 0),
                    }))
                  }
                />
              </label>
              <label className="grid gap-1 text-xs">
                <span className="flex justify-between">
                  <span>{t("chat.maxTokens")}</span>
                  <span className="font-mono">{settings.max_tokens}</span>
                </span>
                <input
                  type="number"
                  min={16}
                  max={Math.max(64, contextLimit)}
                  className="rounded-md border bg-background px-2 py-1 text-xs"
                  value={settings.max_tokens}
                  onChange={(event) =>
                    setSettings((prev) => ({
                      ...prev,
                      max_tokens: Math.max(16, Number(event.target.value) || 128),
                    }))
                  }
                />
              </label>
              <label className="grid gap-1 text-xs">
                <span className="flex justify-between">
                  <span>{t("chat.repetitionPenalty")}</span>
                  <span className="font-mono">{settings.repetition_penalty.toFixed(2)}</span>
                </span>
                <input
                  type="range"
                  min={1}
                  max={1.8}
                  step={0.05}
                  value={settings.repetition_penalty}
                  onChange={(event) =>
                    setSettings((prev) => ({
                      ...prev,
                      repetition_penalty: Number(event.target.value),
                    }))
                  }
                />
              </label>
            </div>
            <div className="mx-auto mt-3 flex max-w-4xl flex-col gap-2 sm:flex-row sm:items-end">
              <label className="grid flex-1 gap-1 text-xs">
                <span>{t("chat.systemPrompt")}</span>
                <textarea
                  rows={2}
                  className="rounded-lg border bg-background p-2 text-xs outline-none focus:ring-1 focus:ring-primary"
                  value={systemPrompt}
                  onChange={(event) => setSystemPrompt(event.target.value)}
                  placeholder={t("chat.systemPromptPlaceholder")}
                />
              </label>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="h-9 text-xs"
                  onClick={() => {
                    setSettings(DEFAULT_SETTINGS);
                    setSystemPrompt("");
                  }}
                >
                  {t("common.reset")}
                </Button>
                {selected ? (
                  <Button
                    type="button"
                    className="h-9 text-xs"
                    disabled={updateChat.isPending}
                    onClick={saveSettings}
                  >
                    {t("common.save")}
                  </Button>
                ) : null}
              </div>
            </div>
          </div>
        ) : null}

        <div
          ref={viewportRef}
          onScroll={handleViewportScroll}
          className="flex-1 overflow-y-auto px-4 py-6 sm:px-8"
        >
          {!selected || (!chat.data?.messages.length && !visibleActive) ? (
            <div className="mx-auto flex max-w-xl flex-col items-center py-16 text-center">
              <div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-primary text-xl font-bold text-primary-foreground shadow-sm">
                A
              </div>
              <h2 className="text-xl font-semibold">{t("chat.startConversation")}</h2>
              <p className="mt-2 text-sm text-muted-foreground">{t("chat.localOnly")}</p>
              <p className="mt-3 rounded-full border bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
                {activeModel
                  ? `${activeModel.model_id} · ${activeModel.checkpoint_id}`
                  : t("chat.noModelLoaded")}
              </p>
              {!selected ? (
                <Button className="mt-5 gap-2" onClick={() => create.mutate()}>
                  <Plus size={16} />
                  {t("chat.newChat")}
                </Button>
              ) : null}
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6">
              {(chat.data?.messages ?? [])
                .filter((message) => message.id !== visibleActive?.messageId || !running)
                .map((message) => {
                  const isUser = message.role === "user";
                  const isEditing = editingMessageId === message.id;
                  const resolvedModelLabel =
                    message.model_id ?? activeModel?.model_id ?? dashboard.data?.inference_model;
                  return (
                    <article
                      key={message.id}
                      className={
                        isUser
                          ? "ml-auto max-w-[85%] rounded-2xl bg-primary px-4 py-3 text-primary-foreground shadow-sm"
                          : "rounded-2xl border bg-background/60 p-4 shadow-sm"
                      }
                    >
                      {!isUser && message.reasoning_trace && message.reasoning_trace.length > 0 ? (
                        <ReasoningPanel events={message.reasoning_trace} />
                      ) : null}

                      {isEditing ? (
                        <div className="grid gap-2">
                          <textarea
                            className="min-h-20 w-full rounded-lg border bg-background p-2 text-sm text-foreground outline-none"
                            value={editingText}
                            onChange={(event) => setEditingText(event.target.value)}
                            aria-label={t("chat.edit")}
                          />
                          <div className="flex justify-end gap-2">
                            <Button
                              type="button"
                              variant="outline"
                              className="h-8 gap-1 px-2.5 text-xs"
                              onClick={() => setEditingMessageId(null)}
                            >
                              <X size={13} />
                              {t("common.cancel")}
                            </Button>
                            <Button
                              type="button"
                              className="h-8 gap-1 px-2.5 text-xs"
                              disabled={!editingText.trim() || running}
                              onClick={() => {
                                const updated = editingText.trim();
                                setEditingMessageId(null);
                                dispatchGeneration({
                                  action: "edit",
                                  content: updated,
                                  targetMessageId: message.id,
                                });
                              }}
                            >
                              <Send size={13} />
                              {t("chat.saveAndSend")}
                            </Button>
                          </div>
                        </div>
                      ) : isUser ? (
                        <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                          {message.content}
                        </div>
                      ) : (
                        <MarkdownContent content={message.content} />
                      )}

                      <div
                        className={`mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] ${
                          isUser ? "text-primary-foreground/80" : "text-muted-foreground"
                        }`}
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          {!isUser && resolvedModelLabel ? (
                            <span className="rounded bg-muted px-1.5 py-0.5 font-medium text-foreground">
                              {resolvedModelLabel}
                            </span>
                          ) : null}
                          {!isUser && message.checkpoint_id ? (
                            <span className="truncate font-mono text-[10px]">
                              {message.checkpoint_id}
                            </span>
                          ) : null}
                          {!isUser && message.tokens_per_second ? (
                            <span className="tabular-nums">
                              {t("chat.tokensPerSecond", {
                                count: Math.round(message.tokens_per_second * 10) / 10,
                              })}
                            </span>
                          ) : null}
                          {!isUser && message.output_tokens ? (
                            <span className="tabular-nums">
                              {t("chat.tokensCount", { count: message.output_tokens })}
                            </span>
                          ) : null}
                          {!isUser && ["cancelled", "interrupted"].includes(message.status) ? (
                            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-amber-600 dark:text-amber-400">
                              {t("chat.partialResponse")}
                            </span>
                          ) : null}
                        </div>

                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 rounded px-1.5 py-1 opacity-75 transition-opacity hover:opacity-100"
                            aria-label={t("chat.copy")}
                            onClick={() => copyText(message.id, message.content)}
                          >
                            {copiedId === message.id ? <Check size={13} /> : <Copy size={13} />}
                            <span>{copiedId === message.id ? t("common.copied") : t("chat.copy")}</span>
                          </button>
                          {isUser && !isEditing ? (
                            <button
                              type="button"
                              className="inline-flex items-center gap-1 rounded px-1.5 py-1 opacity-75 transition-opacity hover:opacity-100"
                              aria-label={t("chat.edit")}
                              disabled={running}
                              onClick={() => {
                                setEditingMessageId(message.id);
                                setEditingText(message.content);
                              }}
                            >
                              <Pencil size={13} />
                              <span>{t("chat.edit")}</span>
                            </button>
                          ) : null}
                          {!isUser ? (
                            <>
                              <button
                                type="button"
                                className="inline-flex items-center gap-1 rounded px-1.5 py-1 opacity-75 transition-opacity hover:opacity-100"
                                aria-label={t("chat.regenerate")}
                                disabled={running}
                                onClick={() =>
                                  dispatchGeneration({
                                    action: "regenerate",
                                    targetMessageId: message.id,
                                  })
                                }
                              >
                                <RefreshCw size={13} />
                                <span>{t("chat.regenerate")}</span>
                              </button>
                              <button
                                type="button"
                                className="inline-flex items-center gap-1 rounded px-1.5 py-1 opacity-75 transition-opacity hover:opacity-100"
                                aria-label={t("chat.continue")}
                                disabled={running}
                                onClick={() =>
                                  dispatchGeneration({
                                    action: "continue",
                                    targetMessageId: message.id,
                                  })
                                }
                              >
                                <FastForward size={13} />
                                <span>{t("chat.continue")}</span>
                              </button>
                            </>
                          ) : null}
                        </div>
                      </div>
                    </article>
                  );
                })}

              {visibleActive && running ? (
                <article className="rounded-2xl border border-primary/30 bg-background/80 p-4 shadow-sm">
                  {visibleActive.status === "queued" ? (
                    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-muted/60 px-3 py-2.5 text-xs">
                      <div className="flex items-center gap-2">
                        <Loader2 size={14} className="animate-spin text-primary" />
                        <div>
                          <p className="font-medium text-foreground">
                            {t("chat.queued", { position: visibleActive.queuePosition ?? 1 })}
                          </p>
                          {visibleActive.queueReason ? (
                            <p className="text-muted-foreground">
                              {t(`chat.queueReasons.${visibleActive.queueReason}`, {
                                defaultValue: visibleActive.queueReason,
                              })}
                            </p>
                          ) : null}
                        </div>
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        className="h-8 gap-1 text-xs"
                        onClick={() =>
                          send({
                            type: "generation.cancel",
                            request_id: visibleActive.requestId,
                          })
                        }
                      >
                        <X size={13} />
                        {t("chat.cancelQueued")}
                      </Button>
                    </div>
                  ) : null}

                  <ReasoningPanel events={visibleActive.reasoningEvents} active />

                  {visibleActive.text ? (
                    <div className="relative">
                      <MarkdownContent content={visibleActive.text} />
                      <span className="ml-0.5 inline-block animate-pulse text-primary">▍</span>
                    </div>
                  ) : null}

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                    <span className="flex items-center gap-2">
                      <Loader2 size={12} className="animate-spin text-primary" />
                      <span>
                        {visibleActive.status === "prefill" || visibleActive.status === "queued"
                          ? t("chat.preparing")
                          : t("chat.generating")}
                      </span>
                      {visibleActive.modelId ? (
                        <span className="rounded bg-muted px-1.5 py-0.5 font-medium text-foreground">
                          {visibleActive.modelId}
                        </span>
                      ) : null}
                      {visibleActive.device ? (
                        <span className="rounded bg-muted px-1.5 py-0.5 font-mono uppercase">
                          {visibleActive.device}
                        </span>
                      ) : null}
                    </span>
                  </div>
                </article>
              ) : null}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {!pinnedToBottom && selected ? (
          <Button
            type="button"
            variant="outline"
            className="absolute bottom-28 right-6 h-9 gap-1.5 rounded-full px-3 text-xs shadow-md"
            onClick={() => scrollToBottom("smooth")}
          >
            <ArrowDown size={14} />
            {t("chat.scrollToBottom")}
          </Button>
        ) : null}

        {selected ? (
          <form className="mx-auto w-full max-w-3xl border-t bg-card px-4 py-3" onSubmit={submit}>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-1" role="radiogroup" aria-label={t("chat.settings")}>
                {MODES.map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    role="radio"
                    aria-checked={settings.mode === mode}
                    className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                      settings.mode === mode
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted/70 text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                    onClick={() => setSettings((prev) => ({ ...prev, mode }))}
                  >
                    {t(`chat.modes.${mode}`)}
                  </button>
                ))}
              </div>
              <span className="text-[11px] text-muted-foreground">
                T={settings.temperature.toFixed(1)} · P={settings.top_p.toFixed(2)} · K={settings.top_k} · Max={settings.max_tokens}
              </span>
            </div>

            <div className="flex items-end gap-2 rounded-xl border bg-background p-2 shadow-sm focus-within:ring-1 focus-within:ring-primary">
              <textarea
                className="max-h-40 min-h-10 flex-1 resize-y bg-transparent p-2 text-sm outline-none"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder={t("chat.prompt")}
                aria-label={t("chat.prompt")}
                rows={1}
              />
              {running && active ? (
                <Button
                  type="button"
                  variant="danger"
                  aria-label={t("chat.stop")}
                  onClick={() =>
                    send({ type: "generation.cancel", request_id: active.requestId })
                  }
                >
                  <Square size={16} />
                </Button>
              ) : (
                <Button
                  type="submit"
                  aria-label={t("chat.send")}
                  disabled={!prompt.trim() || !backendOnline}
                >
                  <Send size={16} />
                </Button>
              )}
            </div>

            {errorCode ? (
              <p role="alert" className="pt-2 text-xs text-destructive">
                {resolveErrorMessage()}
              </p>
            ) : null}
          </form>
        ) : null}
      </section>
    </div>
  );
}
