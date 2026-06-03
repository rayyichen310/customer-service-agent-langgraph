import {
  ArrowLeft,
  Bot,
  CheckCircle2,
  Circle,
  Database,
  GitBranch,
  Loader2,
  MessageSquare,
  Play,
  RefreshCcw,
  RotateCcw,
  Send,
  UserRound,
  XCircle,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { checkHealth, fetchCustomerSnapshot, resetDemoData, streamChat } from "./api";
import type {
  ApiStatus,
  ChatResponse,
  CustomerCandidate,
  CustomerSnapshot,
  MessageRecord,
  StreamNodeEvent,
} from "./types";

const CUSTOMER_CANDIDATES: CustomerCandidate[] = [
  {
    id: 1,
    name: "Alice Chen",
    email: "alice@example.com",
    orders: "12345, 1001, 7890",
  },
  {
    id: 2,
    name: "Bob Lin",
    email: "bob@example.com",
    orders: "5678",
  },
  {
    id: 3,
    name: "Carol Wang",
    email: "carol@example.com",
    orders: "2222",
  },
];

const PROMPTS_BY_CUSTOMER: Record<number, string[]> = {
  1: [
    "Where is my order 12345?",
    "Check status of order 1001",
    "Show my profile",
    "Refund order 7890 if delivered",
    "Cancel it",
    "Remember I prefer refunds",
    "What issues have I had before?",
    "My order is late again",
    "Refund order 0000",
  ],
  2: ["Refund order 5678"],
  3: ["I want to complain about order 2222"],
};

const FALLBACK_PROMPTS = ["Show my profile", "What are my orders?", "Cancel it"];

const READ_KEYS = [
  "order_lookup",
  "order",
  "customer",
  "orders",
  "memories",
  "complaints",
  "issue_patterns",
];

const ACTION_KEYS = ["refund", "cancelled_order", "complaint", "memory_write"];

type FlowNodeId = "planner" | "read_tools" | "verifier" | "actions" | "respond";

const FLOW_NODE_ORDER: FlowNodeId[] = ["planner", "read_tools", "verifier", "actions", "respond"];

type FlowNode = {
  id: FlowNodeId;
  label: string;
  status: "active" | "skipped";
  caption: string;
  steps: FlowStep[];
};

type FlowStep = {
  step: number;
  node: FlowNodeId;
  state: Record<string, unknown>;
};

type FlowEdge = {
  from: FlowStep;
  to: FlowStep;
  label: string;
};

type FlowModel = {
  nodes: FlowNode[];
  steps: FlowStep[];
  edges: FlowEdge[];
};

type StateEdge = {
  from: FlowNodeId;
  to: FlowNodeId;
};

const STATE_EDGES: StateEdge[] = [
  { from: "planner", to: "read_tools" },
  { from: "planner", to: "verifier" },
  { from: "read_tools", to: "planner" },
  { from: "read_tools", to: "verifier" },
  { from: "verifier", to: "planner" },
  { from: "verifier", to: "actions" },
  { from: "verifier", to: "respond" },
  { from: "actions", to: "respond" },
];

const STATE_NODE_LAYOUT: Record<FlowNodeId, { x: number; y: number }> = {
  planner: { x: 96, y: 58 },
  read_tools: { x: 320, y: 58 },
  verifier: { x: 210, y: 168 },
  actions: { x: 430, y: 168 },
  respond: { x: 430, y: 268 },
};

function createId(prefix: string): string {
  if (crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [activeCustomerId, setActiveCustomerId] = useState<number | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [input, setInput] = useState("");
  const [manualCustomerId, setManualCustomerId] = useState("");
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [mobileTab, setMobileTab] = useState<"chat" | "inspector">("chat");
  const [liveNodeTrace, setLiveNodeTrace] = useState<StreamNodeEvent[]>([]);
  const [isLivePlaybackActive, setIsLivePlaybackActive] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<"flow" | "database">("flow");
  const [customerSnapshot, setCustomerSnapshot] = useState<CustomerSnapshot | null>(null);
  const [isSnapshotLoading, setIsSnapshotLoading] = useState(false);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function refreshHealth() {
      try {
        await checkHealth();
        if (!cancelled) {
          setApiStatus("online");
        }
      } catch {
        if (!cancelled) {
          setApiStatus("offline");
        }
      }
    }

    refreshHealth();
    const interval = window.setInterval(refreshHealth, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  const inspectorState = useMemo(() => {
    if (isLivePlaybackActive) {
      return {
        payload: null,
        nodeTrace: liveNodeTrace,
      };
    }

    const selected = messages.find((message) => message.id === selectedMessageId);
    if (selected?.payload) {
      return {
        payload: selected.payload,
        nodeTrace: selected.nodeTrace ?? [],
      };
    }

    const latest = [...messages].reverse().find((message) => message.payload);
    return {
      payload: latest?.payload ?? null,
      nodeTrace: latest?.nodeTrace ?? [],
    };
  }, [isLivePlaybackActive, liveNodeTrace, messages, selectedMessageId]);

  const activeCustomer = CUSTOMER_CANDIDATES.find((customer) => customer.id === activeCustomerId);

  async function refreshCustomerSnapshot(customerId: number) {
    setIsSnapshotLoading(true);
    setSnapshotError(null);
    try {
      const snapshot = await fetchCustomerSnapshot(customerId);
      setCustomerSnapshot(snapshot);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Database snapshot failed";
      setSnapshotError(message);
    } finally {
      setIsSnapshotLoading(false);
    }
  }

  function startThread(customerId: number) {
    setActiveCustomerId(customerId);
    setThreadId(createId("thread"));
    setMessages([]);
    setSelectedMessageId(null);
    setLiveNodeTrace([]);
    setIsLivePlaybackActive(false);
    setInput("");
    setNotice(null);
    setMobileTab("chat");
    setInspectorTab("flow");
    setCustomerSnapshot(null);
    setSnapshotError(null);
    void refreshCustomerSnapshot(customerId);
  }

  function exitThread() {
    setActiveCustomerId(null);
    setThreadId(null);
    setMessages([]);
    setSelectedMessageId(null);
    setLiveNodeTrace([]);
    setIsLivePlaybackActive(false);
    setInput("");
    setMobileTab("chat");
    setInspectorTab("flow");
    setCustomerSnapshot(null);
    setSnapshotError(null);
  }

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || !threadId || activeCustomerId === null || isSending || apiStatus === "offline") {
      return;
    }

    const userMessage: MessageRecord = {
      id: createId("user"),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    };
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setIsSending(true);
    setNotice(null);
    setLiveNodeTrace([]);
    setIsLivePlaybackActive(true);
    setSelectedMessageId(null);
    setMobileTab("inspector");

    let replayTimer: number | undefined;
    let queuedNodeTrace: StreamNodeEvent[] = [];
    let revealedNodeTrace: StreamNodeEvent[] = [];
    let streamCompleted = false;

    const finishLivePlaybackIfReady = () => {
      if (!streamCompleted || queuedNodeTrace.length > 0 || replayTimer !== undefined) {
        return;
      }
      setLiveNodeTrace([]);
      setIsLivePlaybackActive(false);
    };

    const scheduleReveal = () => {
      if (replayTimer !== undefined) {
        return;
      }
      replayTimer = window.setTimeout(() => {
        replayTimer = undefined;
        const next = queuedNodeTrace.shift();
        if (!next) {
          finishLivePlaybackIfReady();
          return;
        }
        revealedNodeTrace = [...revealedNodeTrace, next];
        setLiveNodeTrace(revealedNodeTrace);
        scheduleReveal();
      }, 850);
    };

    const stopLivePlayback = () => {
      if (replayTimer !== undefined) {
        window.clearTimeout(replayTimer);
        replayTimer = undefined;
      }
      queuedNodeTrace = [];
      revealedNodeTrace = [];
      setLiveNodeTrace([]);
      setIsLivePlaybackActive(false);
    };

    try {
      const streamResult: {
        nodeTrace: StreamNodeEvent[];
        payload?: ChatResponse;
        error?: string;
      } = {
        nodeTrace: [],
      };

      await streamChat(
        {
          thread_id: threadId,
          message: trimmed,
          customer_id: activeCustomerId,
        },
        (event) => {
          if (event.event === "started") {
            streamResult.nodeTrace = [];
            queuedNodeTrace = [];
            revealedNodeTrace = [];
            setLiveNodeTrace([]);
          }
          if (event.event === "node") {
            streamResult.nodeTrace = [...streamResult.nodeTrace, event.data];
            queuedNodeTrace.push(event.data);
            scheduleReveal();
          }
          if (event.event === "final") {
            streamResult.payload = event.data;
          }
          if (event.event === "error") {
            streamResult.error = event.data.message;
          }
        },
      );

      if (streamResult.error) {
        throw new Error(streamResult.error);
      }
      if (!streamResult.payload) {
        throw new Error("Stream ended without a final response.");
      }
      const payload = streamResult.payload;

      const assistantMessage: MessageRecord = {
        id: createId("assistant"),
        role: "assistant",
        content: payload.response || "(empty response)",
        createdAt: new Date().toISOString(),
        payload,
        nodeTrace: streamResult.nodeTrace,
      };
      setMessages((current) => [...current, assistantMessage]);
      setSelectedMessageId(assistantMessage.id);
      setMobileTab("chat");
      streamCompleted = true;
      finishLivePlaybackIfReady();
      void refreshCustomerSnapshot(activeCustomerId);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat request failed";
      setMessages((current) => [
        ...current,
        {
          id: createId("error"),
          role: "error",
          content: message,
          createdAt: new Date().toISOString(),
        },
      ]);
      stopLivePlayback();
    } finally {
      setIsSending(false);
    }
  }

  async function handleResetConfirmed() {
    setIsResetting(true);
    setNotice(null);
    try {
      await resetDemoData();
      setResetDialogOpen(false);
      exitThread();
      setNotice("Demo data reset.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Initialize failed";
      setNotice(message);
    } finally {
      setIsResetting(false);
    }
  }

  if (activeCustomerId === null || threadId === null) {
    return (
      <main className="app-shell selection-shell">
        <TopBar apiStatus={apiStatus} onInitialize={() => setResetDialogOpen(true)} />
        <section className="selection-layout">
          <div className="section-header">
            <h1>Customer selection</h1>
            <p>Choose a seeded customer or enter a customer ID.</p>
          </div>

          {notice && <StatusMessage message={notice} tone={notice.includes("reset") ? "ok" : "error"} />}

          <div className="customer-grid">
            {CUSTOMER_CANDIDATES.map((customer) => (
              <button
                className="customer-card"
                key={customer.id}
                onClick={() => startThread(customer.id)}
                type="button"
              >
                <span className="customer-avatar">
                  <UserRound size={20} />
                </span>
                <span>
                  <strong>
                    {customer.id} - {customer.name}
                  </strong>
                  <small>{customer.email}</small>
                  <small>Orders {customer.orders}</small>
                </span>
              </button>
            ))}
          </div>

          <form
            className="manual-entry"
            onSubmit={(event) => {
              event.preventDefault();
              const customerId = Number.parseInt(manualCustomerId, 10);
              if (Number.isFinite(customerId)) {
                startThread(customerId);
              }
            }}
          >
            <label htmlFor="manual-customer-id">Customer ID</label>
            <div className="inline-controls">
              <input
                id="manual-customer-id"
                inputMode="numeric"
                onChange={(event) => setManualCustomerId(event.target.value)}
                placeholder="Enter customer ID"
                value={manualCustomerId}
              />
              <button className="primary-button" type="submit">
                <Play size={16} />
                Start
              </button>
            </div>
          </form>
        </section>

        <ResetDialog
          isOpen={resetDialogOpen}
          isResetting={isResetting}
          onCancel={() => setResetDialogOpen(false)}
          onConfirm={handleResetConfirmed}
        />
      </main>
    );
  }

  return (
    <main className="app-shell workspace-shell">
      <TopBar apiStatus={apiStatus} onInitialize={() => setResetDialogOpen(true)} />

      <header className="thread-bar">
        <button className="ghost-button" onClick={exitThread} type="button">
          <ArrowLeft size={16} />
          Exit thread
        </button>
        <div>
          <strong>
            Customer {activeCustomerId}
            {activeCustomer ? ` - ${activeCustomer.name}` : ""}
          </strong>
          <span>{threadId}</span>
        </div>
      </header>

      {notice && <StatusMessage message={notice} tone={notice.includes("failed") ? "error" : "ok"} />}

      <div className="mobile-tabs">
        <button
          className={mobileTab === "chat" ? "tab-button active" : "tab-button"}
          onClick={() => setMobileTab("chat")}
          type="button"
        >
          Chat
        </button>
        <button
          className={mobileTab === "inspector" ? "tab-button active" : "tab-button"}
          onClick={() => setMobileTab("inspector")}
          type="button"
        >
          Inspector
        </button>
      </div>

      <section className="workspace-grid">
        <div className={mobileTab === "chat" ? "chat-panel visible-mobile-panel" : "chat-panel"}>
          <PromptShortcuts customerId={activeCustomerId} onPick={setInput} />
          <MessageList
            isSending={isSending}
            messages={messages}
            onSelect={(messageId) => {
              setIsLivePlaybackActive(false);
              setSelectedMessageId(messageId);
            }}
            selectedMessageId={selectedMessageId}
          />
          <form className="composer" onSubmit={handleSend}>
            <textarea
              aria-label="Message"
              disabled={isSending}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Type a customer-service request"
              rows={3}
              value={input}
            />
            <button
              className="primary-button send-button"
              disabled={!input.trim() || isSending || apiStatus === "offline"}
              type="submit"
            >
              {isSending ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              Send
            </button>
          </form>
        </div>

        <div
          className={
            mobileTab === "inspector"
              ? "inspector-panel visible-mobile-panel"
              : "inspector-panel"
          }
        >
          <InspectorShell
            activeTab={inspectorTab}
            customerId={activeCustomerId}
            isSnapshotLoading={isSnapshotLoading}
            nodeTrace={inspectorState.nodeTrace}
            onRefreshSnapshot={() => refreshCustomerSnapshot(activeCustomerId)}
            onTabChange={setInspectorTab}
            payload={inspectorState.payload}
            snapshot={customerSnapshot}
            snapshotError={snapshotError}
          />
        </div>
      </section>

      <ResetDialog
        isOpen={resetDialogOpen}
        isResetting={isResetting}
        onCancel={() => setResetDialogOpen(false)}
        onConfirm={handleResetConfirmed}
      />
    </main>
  );
}

function TopBar({
  apiStatus,
  onInitialize,
}: {
  apiStatus: ApiStatus;
  onInitialize: () => void;
}) {
  return (
    <header className="top-bar">
      <div className="brand">
        <Bot size={22} />
        <span>Customer Service Agent</span>
      </div>
      <div className="top-actions">
        <StatusBadge status={apiStatus} />
        <button className="secondary-button" onClick={onInitialize} type="button">
          <RefreshCcw size={16} />
          Initialize
        </button>
      </div>
    </header>
  );
}

function StatusBadge({ status }: { status: ApiStatus }) {
  const label = status === "checking" ? "Checking" : status === "online" ? "API online" : "API offline";
  return (
    <span className={`status-badge ${status}`}>
      {status === "online" ? <CheckCircle2 size={14} /> : <Circle size={14} />}
      {label}
    </span>
  );
}

function StatusMessage({ message, tone }: { message: string; tone: "ok" | "error" }) {
  return <div className={`status-message ${tone}`}>{message}</div>;
}

function PromptShortcuts({
  customerId,
  onPick,
}: {
  customerId: number;
  onPick: (prompt: string) => void;
}) {
  const prompts = PROMPTS_BY_CUSTOMER[customerId] ?? FALLBACK_PROMPTS;

  return (
    <div className="prompt-strip">
      {prompts.map((prompt) => (
        <button key={prompt} onClick={() => onPick(prompt)} type="button">
          <MessageSquare size={14} />
          {prompt}
        </button>
      ))}
    </div>
  );
}

function MessageList({
  isSending,
  messages,
  onSelect,
  selectedMessageId,
}: {
  isSending: boolean;
  messages: MessageRecord[];
  onSelect: (messageId: string) => void;
  selectedMessageId: string | null;
}) {
  return (
    <div className="message-list">
      {messages.length === 0 && (
        <div className="empty-state">
          <Bot size={28} />
          <span>No messages in this thread.</span>
        </div>
      )}
      {messages.map((message) => (
        <button
          className={`message-row ${message.role} ${
            selectedMessageId === message.id ? "selected" : ""
          }`}
          key={message.id}
          onClick={() => {
            if (message.payload) {
              onSelect(message.id);
            }
          }}
          type="button"
        >
          <span className="message-icon">
            {message.role === "user" ? (
              <UserRound size={16} />
            ) : message.role === "assistant" ? (
              <Bot size={16} />
            ) : (
              <XCircle size={16} />
            )}
          </span>
          <span className="message-body">
            <span>{message.content}</span>
            {message.payload && <DecisionChip payload={message.payload} />}
          </span>
        </button>
      ))}
      {isSending && (
        <div className="thinking-row">
          <Loader2 className="spin" size={16} />
          Agent is thinking...
        </div>
      )}
    </div>
  );
}

function DecisionChip({ payload }: { payload: ChatResponse }) {
  const decision = String(payload.verification_decision.decision ?? "response");
  return <small className={`decision-chip ${decision}`}>{decision}</small>;
}

function InspectorShell({
  activeTab,
  customerId,
  isSnapshotLoading,
  nodeTrace,
  onRefreshSnapshot,
  onTabChange,
  payload,
  snapshot,
  snapshotError,
}: {
  activeTab: "flow" | "database";
  customerId: number;
  isSnapshotLoading: boolean;
  nodeTrace: StreamNodeEvent[];
  onRefreshSnapshot: () => void;
  onTabChange: (tab: "flow" | "database") => void;
  payload: ChatResponse | null;
  snapshot: CustomerSnapshot | null;
  snapshotError: string | null;
}) {
  return (
    <div className="inspector-shell">
      <div className="inspector-tabs" role="tablist" aria-label="Inspector views">
        <button
          className={activeTab === "flow" ? "tab-button active" : "tab-button"}
          onClick={() => onTabChange("flow")}
          type="button"
        >
          <GitBranch size={14} />
          Flow
        </button>
        <button
          className={activeTab === "database" ? "tab-button active" : "tab-button"}
          onClick={() => onTabChange("database")}
          type="button"
        >
          <Database size={14} />
          Database
        </button>
      </div>

      {activeTab === "flow" ? (
        <Inspector nodeTrace={nodeTrace} payload={payload} />
      ) : (
        <DatabasePanel
          customerId={customerId}
          isLoading={isSnapshotLoading}
          onRefresh={onRefreshSnapshot}
          snapshot={snapshot}
          snapshotError={snapshotError}
        />
      )}
    </div>
  );
}

function DatabasePanel({
  customerId,
  isLoading,
  onRefresh,
  snapshot,
  snapshotError,
}: {
  customerId: number;
  isLoading: boolean;
  onRefresh: () => void;
  snapshot: CustomerSnapshot | null;
  snapshotError: string | null;
}) {
  return (
    <div className="database-panel">
      <div className="panel-title database-title">
        <div>
          <h2>Customer database</h2>
          <small>customer_id {customerId}</small>
        </div>
        <button className="icon-button" disabled={isLoading} onClick={onRefresh} type="button">
          {isLoading ? <Loader2 className="spin" size={15} /> : <RefreshCcw size={15} />}
        </button>
      </div>

      {snapshotError && <StatusMessage message={snapshotError} tone="error" />}

      {!snapshot && !snapshotError && (
        <div className="inspector-empty database-empty">
          {isLoading ? <Loader2 className="spin" size={24} /> : <Database size={26} />}
          <span>{isLoading ? "Loading customer rows..." : "No database snapshot loaded."}</span>
        </div>
      )}

      {snapshot && (
        <div className="database-sections">
          <KeyValueSection title="Customer" value={snapshot.customer} />
          <RecordTable
            columns={["order_id", "product_name", "status", "order_date", "delivery_date"]}
            emptyText="No orders for this customer."
            records={snapshot.orders}
            title="Orders"
          />
          <RecordTable
            columns={["complaint_id", "order_id", "issue", "status", "created_at"]}
            emptyText="No complaints for this customer."
            records={snapshot.complaints}
            title="Complaints"
          />
          <RecordTable
            columns={["key", "value", "created_at"]}
            emptyText="No memory rows for this customer."
            records={snapshot.memories}
            title="Memory"
          />
          <KeyValueSection title="Issue patterns" value={snapshot.issue_patterns} />
        </div>
      )}
    </div>
  );
}

function RecordTable({
  columns,
  emptyText,
  records,
  title,
}: {
  columns: string[];
  emptyText: string;
  records: Array<Record<string, unknown>>;
  title: string;
}) {
  return (
    <section className="database-section">
      <h3>{title}</h3>
      {records.length === 0 ? (
        <p className="database-muted">{emptyText}</p>
      ) : (
        <div className="database-table-wrap">
          <table className="database-table">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((record, index) => (
                <tr key={`${title}-${index}`}>
                  {columns.map((column) => (
                    <td key={column}>{formatDatabaseValue(record[column])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function KeyValueSection({
  title,
  value,
}: {
  title: string;
  value: Record<string, unknown> | null;
}) {
  const entries = Object.entries(value ?? {});
  return (
    <section className="database-section">
      <h3>{title}</h3>
      {entries.length === 0 ? (
        <p className="database-muted">No data.</p>
      ) : (
        <dl className="database-kv">
          {entries.map(([key, item]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{formatDatabaseValue(item)}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

function Inspector({
  nodeTrace,
  payload,
}: {
  nodeTrace: StreamNodeEvent[];
  payload: ChatResponse | null;
}) {
  const [selectedNodeId, setSelectedNodeId] = useState<FlowNodeId>("verifier");
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [replayStep, setReplayStep] = useState(0);
  const [replayRun, setReplayRun] = useState(0);

  const flow = useMemo(() => buildFlowModel(payload, nodeTrace), [nodeTrace, payload]);

  useEffect(() => {
    const latestStep = flow.steps.at(-1);
    if (!latestStep) {
      return;
    }

    if (!payload) {
      setSelectedNodeId(latestStep.node);
      setSelectedStep(latestStep.step);
      return;
    }

    const verifierStep = latestStepForNode(flow, "verifier") ?? latestStep;
    setSelectedNodeId(verifierStep.node);
    setSelectedStep(verifierStep.step);
  }, [flow, nodeTrace.length, payload]);

  useEffect(() => {
    if (flow.steps.length === 0) {
      return;
    }

    if (!payload) {
      setReplayStep(flow.steps.length);
      return;
    }

    if (replayRun === 0) {
      setReplayStep(flow.steps.length);
      return;
    }

    setReplayStep(0);

    let step = 0;
    const interval = window.setInterval(() => {
      step += 1;
      setReplayStep(step);
      if (step >= flow.steps.length) {
        window.clearInterval(interval);
      }
    }, 850);

    return () => window.clearInterval(interval);
  }, [flow.steps.length, payload, replayRun]);

  if (flow.steps.length === 0) {
    return (
      <div className="inspector-empty">
        <Database size={26} />
        <span>No assistant output selected.</span>
      </div>
    );
  }

  const selectedNode = flow.nodes.find((node) => node.id === selectedNodeId) ?? flow.nodes[0];
  const selectedFlowStep =
    selectedNode.steps.find((step) => step.step === selectedStep) ?? selectedNode.steps.at(-1) ?? null;

  return (
    <div className="flow-inspector">
      <section className="flow-card">
        <div className="panel-title">
          <div>
          <h2>Execution flow</h2>
            <small>{payload ? "replay from saved node trace" : "live node trace"}</small>
          </div>
          <button
            aria-label="Replay current flow"
            className="icon-button"
            disabled={!payload}
            onClick={() => setReplayRun((current) => current + 1)}
            type="button"
          >
            <RotateCcw size={15} />
          </button>
        </div>
        <FlowChart
          flow={flow}
          onSelect={(nodeId) => {
            const node = flow.nodes.find((candidate) => candidate.id === nodeId);
            setSelectedNodeId(nodeId);
            setSelectedStep(node?.steps.at(-1)?.step ?? null);
          }}
          replayStep={replayStep}
          selectedNodeId={selectedNode.id}
        />
      </section>
      <NodeDetail
        node={selectedNode}
        onStepSelect={setSelectedStep}
        payload={payload}
        selectedStep={selectedFlowStep}
      />
    </div>
  );
}

function buildFlowModel(payload: ChatResponse | null, nodeTrace: StreamNodeEvent[]): FlowModel {
  const steps = normalizeTraceSteps(payload, nodeTrace);
  const nodes = FLOW_NODE_ORDER.map((nodeId) => {
    const nodeSteps = steps.filter((step) => step.node === nodeId);
    return {
      id: nodeId,
      label: nodeLabel(nodeId),
      status: nodeSteps.length > 0 ? "active" : "skipped",
      caption: nodeSteps.length > 0 ? `steps ${nodeSteps.map((step) => step.step).join(", ")}` : "not visited",
      steps: nodeSteps,
    } satisfies FlowNode;
  });
  const edges = steps.slice(1).map((step, index) => {
    const from = steps[index];
    return {
      from,
      to: step,
      label: transitionLabel(from, step),
    };
  });

  return { nodes, steps, edges };
}

function FlowChart({
  flow,
  onSelect,
  replayStep,
  selectedNodeId,
}: {
  flow: FlowModel;
  onSelect: (nodeId: FlowNodeId) => void;
  replayStep: number;
  selectedNodeId: FlowNodeId;
}) {
  const activeStep = flow.steps.find((step) => step.step === replayStep) ?? null;
  const activeEdge = flow.edges.find((edge) => edge.to.step === replayStep) ?? null;
  const visibleEdges = groupVisibleEdges(flow.edges, replayStep);

  return (
    <div className="state-machine" aria-label="Dynamic execution state machine">
      <svg className="state-machine-svg" viewBox="0 0 560 330" aria-hidden="true">
        <defs>
          <marker id="state-arrow" markerHeight="8" markerWidth="8" orient="auto" refX="7" refY="4">
            <path d="M0,0 L8,4 L0,8 Z" />
          </marker>
          <marker
            id="state-arrow-active"
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="7"
            refY="4"
          >
            <path d="M0,0 L8,4 L0,8 Z" />
          </marker>
        </defs>
        {visibleEdges.map((edge) => {
          const isActive = edge.edges.some((candidate) => candidate === activeEdge);
          return (
            <g key={`${edge.from}-${edge.to}`}>
              <path
                className={isActive ? "state-edge active" : "state-edge"}
                d={stateEdgePath(edge)}
                markerEnd={isActive ? "url(#state-arrow-active)" : "url(#state-arrow)"}
              />
              <text
                className={isActive ? "state-edge-label active" : "state-edge-label"}
                x={stateEdgeLabelPosition(edge).x}
                y={stateEdgeLabelPosition(edge).y}
              >
                {edgeLabel(edge)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="state-machine-nodes">
        {flow.nodes.map((node) => {
          const revealedSteps = node.steps.filter((step) => step.step <= replayStep);
          const isRevealed = revealedSteps.length > 0;
          const isRunning = activeStep?.node === node.id;
          const position = STATE_NODE_LAYOUT[node.id];
          return (
            <button
              className={`state-node ${node.status} ${selectedNodeId === node.id ? "selected" : ""} ${
                isRevealed ? "revealed" : "pending"
              } ${isRunning ? "running" : ""}`}
              data-node={node.id}
              key={node.id}
              onClick={() => onSelect(node.id)}
              style={{
                left: `${(position.x / 560) * 100}%`,
                top: `${(position.y / 330) * 100}%`,
              }}
              type="button"
            >
              <span className="state-node-icon">
                <GitBranch size={15} />
              </span>
              <span>
                <strong>{node.label}</strong>
                <small>{nodeVisitLabel(revealedSteps.length, isRunning)}</small>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function NodeDetail({
  node,
  onStepSelect,
  payload,
  selectedStep,
}: {
  node: FlowNode;
  onStepSelect: (step: number) => void;
  payload: ChatResponse | null;
  selectedStep: FlowStep | null;
}) {
  const raw = selectedStep?.state ?? {};
  return (
    <section className="node-detail">
      <div className="panel-title">
        <h2>{node.label}</h2>
        <small>{selectedStep ? `step ${selectedStep.step}` : node.status}</small>
      </div>
      {node.steps.length > 1 && (
        <div className="step-selector">
          {node.steps.map((step) => (
            <button
              className={selectedStep?.step === step.step ? "active" : ""}
              key={step.step}
              onClick={() => onStepSelect(step.step)}
              type="button"
            >
              step {step.step}
            </button>
          ))}
        </div>
      )}
      <NodeSummary node={node} payload={payload} raw={raw} />
      <details className="raw-details">
        <summary>Raw JSON</summary>
        <JsonBlock value={raw} />
      </details>
    </section>
  );
}

function NodeSummary({
  node,
  payload,
  raw,
}: {
  node: FlowNode;
  payload: ChatResponse | null;
  raw: Record<string, unknown>;
}) {
  if (!node.steps.length) {
    return (
      <dl className="summary-grid">
        <SummaryItem label="Status" value="Not visited in this execution path" />
      </dl>
    );
  }

  if (node.id === "planner") {
    return (
      <dl className="summary-grid">
        <SummaryItem label="Tool calls" value={nameList(raw.tool_calls)} />
        <SummaryItem label="Requested actions" value={nameList(raw.requested_actions)} />
        <SummaryItem label="Customer" value={stringValue(raw.authenticated_customer_id) ?? "-"} />
        <SummaryItem label="Turn history" value={stringValue(raw.turn_history_count) ?? "0"} />
      </dl>
    );
  }

  if (node.id === "verifier") {
    const decision = asRecord(raw.verification_decision ?? payload?.verification_decision);
    return (
      <dl className="summary-grid">
        <SummaryItem label="Decision" value={String(decision.decision ?? "proceed_to_response")} />
        <SummaryItem label="Reason" value={String(decision.reason_code ?? "-")} />
        <SummaryItem label="Missing slots" value={formatList(decision.missing_slots)} />
        <SummaryItem label="Policy errors" value={formatList(decision.policy_errors)} />
      </dl>
    );
  }

  if (node.id === "respond") {
    const response = payload?.response ?? stringValue(raw.final_response) ?? "-";
    const verifiedFacts = asRecord(raw.verified_facts ?? payload?.verified_facts);
    return (
      <dl className="summary-grid">
        <SummaryItem label="Response" value={compactText(response)} />
        <SummaryItem label="Verified facts" value={Object.keys(verifiedFacts).join(", ") || "-"} />
      </dl>
    );
  }

  if (node.id === "read_tools" || node.id === "actions") {
    const results = asRecord(raw.tool_results ?? raw);
    return (
      <dl className="summary-grid">
        {Object.entries(results).map(([key, value]) => (
          <SummaryItem key={key} label={key} value={summarizeValue(value)} />
        ))}
        {Object.keys(results).length === 0 && (
          <SummaryItem label="Result" value="Skipped for this response" />
        )}
      </dl>
    );
  }

  return (
    <dl className="summary-grid">
      <SummaryItem label="Status" value="Visited" />
    </dl>
  );
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

function formatList(value: unknown) {
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => JSON.stringify(item)).join(", ") : "-";
  }
  return "-";
}

function formatDatabaseValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function normalizeTraceSteps(
  payload: ChatResponse | null,
  nodeTrace: StreamNodeEvent[],
): FlowStep[] {
  const traceSteps = nodeTrace
    .filter((event) => isFlowNodeId(event.node))
    .map((event, index) => ({
      step: index + 1,
      node: event.node as FlowNodeId,
      state: event.state,
    }));

  if (traceSteps.length > 0 || !payload) {
    return traceSteps;
  }

  return synthesizeTraceSteps(payload);
}

function synthesizeTraceSteps(payload: ChatResponse): FlowStep[] {
  const steps: Array<Omit<FlowStep, "step">> = [{ node: "planner", state: {} }];
  const toolKeys = Object.keys(payload.tool_results);
  if (toolKeys.some((key) => READ_KEYS.includes(key))) {
    steps.push({
      node: "read_tools",
      state: { tool_results: pickKeys(payload.tool_results, READ_KEYS) },
    });
  }
  steps.push({
    node: "verifier",
    state: { verification_decision: payload.verification_decision },
  });
  if (toolKeys.some((key) => ACTION_KEYS.includes(key))) {
    steps.push({
      node: "actions",
      state: { tool_results: pickKeys(payload.tool_results, ACTION_KEYS) },
    });
  }
  steps.push({
    node: "respond",
    state: {
      final_response: payload.response,
      verified_facts: payload.verified_facts,
    },
  });

  return steps.map((step, index) => ({ ...step, step: index + 1 }));
}

function latestStepForNode(flow: FlowModel, nodeId: FlowNodeId) {
  return flow.nodes.find((node) => node.id === nodeId)?.steps.at(-1) ?? null;
}

function nodeLabel(nodeId: FlowNodeId) {
  const labels: Record<FlowNodeId, string> = {
    planner: "Planner",
    read_tools: "Read tools",
    verifier: "Verifier",
    actions: "Actions",
    respond: "Respond",
  };
  return labels[nodeId];
}

function nodeVisitLabel(visitCount: number, isRunning: boolean) {
  if (isRunning) {
    return "running";
  }
  if (visitCount === 0) {
    return "idle";
  }
  if (visitCount === 1) {
    return "visited";
  }
  return `x${visitCount}`;
}

function groupVisibleEdges(edges: FlowEdge[], replayStep: number) {
  const visible = edges.filter((edge) => edge.to.step <= replayStep);
  return visible.reduce<
    Array<StateEdge & { label: string; edges: FlowEdge[]; orderSteps: number[] }>
  >((groups, edge) => {
    const existing = groups.find(
      (group) => group.from === edge.from.node && group.to === edge.to.node,
    );
    if (existing) {
      existing.edges.push(edge);
      existing.orderSteps.push(edge.from.step);
      return groups;
    }

    return [
      ...groups,
      {
        from: edge.from.node,
        to: edge.to.node,
        label: edge.label,
        edges: [edge],
        orderSteps: [edge.from.step],
      },
    ];
  }, []);
}

function edgeLabel(edge: { label: string; orderSteps: number[] }) {
  return `${edge.label} ${edge.orderSteps.join(",")}`;
}

function stateEdgePath(edge: StateEdge) {
  const from = STATE_NODE_LAYOUT[edge.from];
  const to = STATE_NODE_LAYOUT[edge.to];

  if (edge.from === "read_tools" && edge.to === "planner") {
    return curvePath(from, to, 0, -58);
  }
  if (edge.from === "verifier" && edge.to === "planner") {
    return curvePath(from, to, -76, 0);
  }
  if (edge.from === "planner" && edge.to === "verifier") {
    return curvePath(from, to, -36, 26);
  }
  if (edge.from === "read_tools" && edge.to === "verifier") {
    return curvePath(from, to, 36, 26);
  }
  if (edge.from === "verifier" && edge.to === "respond") {
    return curvePath(from, to, 44, 64);
  }
  return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
}

function curvePath(
  from: { x: number; y: number },
  to: { x: number; y: number },
  offsetX: number,
  offsetY: number,
) {
  const midX = (from.x + to.x) / 2 + offsetX;
  const midY = (from.y + to.y) / 2 + offsetY;
  return `M ${from.x} ${from.y} Q ${midX} ${midY} ${to.x} ${to.y}`;
}

function stateEdgeLabelPosition(edge: StateEdge) {
  const from = STATE_NODE_LAYOUT[edge.from];
  const to = STATE_NODE_LAYOUT[edge.to];
  const midpoint = {
    x: (from.x + to.x) / 2,
    y: (from.y + to.y) / 2 - 10,
  };

  if (edge.from === "read_tools" && edge.to === "planner") {
    return { x: midpoint.x, y: midpoint.y - 42 };
  }
  if (edge.from === "verifier" && edge.to === "planner") {
    return { x: midpoint.x - 52, y: midpoint.y };
  }
  if (edge.from === "verifier" && edge.to === "respond") {
    return { x: midpoint.x + 42, y: midpoint.y + 18 };
  }
  if (edge.from === "verifier" && edge.to === "actions") {
    return { x: midpoint.x, y: midpoint.y - 12 };
  }
  if (edge.from === "planner" && edge.to === "verifier") {
    return { x: midpoint.x - 24, y: midpoint.y + 20 };
  }
  if (edge.from === "read_tools" && edge.to === "verifier") {
    return { x: midpoint.x + 24, y: midpoint.y + 20 };
  }

  return midpoint;
}

function transitionLabel(from: FlowStep, to: FlowStep) {
  if (from.node === "planner" && to.node === "read_tools") {
    return "tool calls";
  }
  if (from.node === "planner" && to.node === "verifier") {
    return "plan";
  }
  if (from.node === "read_tools" && to.node === "planner") {
    return "observations";
  }
  if (from.node === "read_tools" && to.node === "verifier") {
    return "tool results";
  }
  if (from.node === "verifier" && to.node === "planner") {
    return "replan feedback";
  }
  if (from.node === "verifier" && to.node === "actions") {
    return "approved actions";
  }
  if (from.node === "verifier" && to.node === "respond") {
    return "decision";
  }
  if (from.node === "actions" && to.node === "respond") {
    return "action results";
  }
  return "state update";
}

function nameList(value: unknown) {
  if (!Array.isArray(value) || value.length === 0) {
    return "-";
  }
  return value
    .map((item) => {
      const record = asRecord(item);
      return stringValue(record.name) ?? stringValue(record.tool) ?? stringValue(record.action) ?? "item";
    })
    .join(", ");
}

function compactText(value: string) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= 120) {
    return normalized || "-";
  }
  return `${normalized.slice(0, 117)}...`;
}

function pickKeys(source: Record<string, unknown>, keys: string[]) {
  return keys.reduce<Record<string, unknown>>((picked, key) => {
    if (key in source) {
      picked[key] = source[key];
    }
    return picked;
  }, {});
}

function plannerCaption(
  readKeys: string[],
  actionKeys: string[],
  plannerState: Record<string, unknown>,
) {
  const plannedReads = Array.isArray(plannerState.tool_calls) ? plannerState.tool_calls.length : 0;
  const plannedActions = Array.isArray(plannerState.requested_actions)
    ? plannerState.requested_actions.length
    : 0;
  if (plannedReads > 0 && plannedActions > 0) {
    return `${plannedReads} read, ${plannedActions} action`;
  }
  if (plannedActions > 0) {
    return `${plannedActions} action proposal${plannedActions === 1 ? "" : "s"}`;
  }
  if (plannedReads > 0) {
    return `${plannedReads} read call${plannedReads === 1 ? "" : "s"}`;
  }
  if (actionKeys.length > 0) {
    return "Likely action plan";
  }
  if (readKeys.length > 0) {
    return "Likely read plan";
  }
  return "Direct response plan";
}

function readCaption(toolResults: Record<string, unknown>, readKeys: string[]) {
  if (!readKeys.length) {
    return "Skipped";
  }

  const order = asRecord(toolResults.order);
  const orderLookup = asRecord(toolResults.order_lookup);
  if (order.order_id || orderLookup.order_id) {
    const orderId = stringValue(order.order_id ?? orderLookup.order_id);
    const found = orderLookup.found === false ? "not found" : "found";
    return orderId ? `Order ${orderId} ${found}` : "Order read";
  }

  if ("orders" in toolResults && Array.isArray(toolResults.orders)) {
    return `${toolResults.orders.length} orders`;
  }
  if ("customer" in toolResults) {
    return "Customer profile";
  }
  if ("memories" in toolResults && Array.isArray(toolResults.memories)) {
    return `${toolResults.memories.length} memories`;
  }
  if ("complaints" in toolResults && Array.isArray(toolResults.complaints)) {
    return `${toolResults.complaints.length} complaints`;
  }
  if ("issue_patterns" in toolResults) {
    return "Issue patterns";
  }

  return "Read result";
}

function verifierCaption(decision: Record<string, unknown>) {
  const decisionText = stringValue(decision.decision) ?? "proceed_to_response";
  const missing = Array.isArray(decision.missing_slots) ? decision.missing_slots.length : 0;
  const policy = Array.isArray(decision.policy_errors) ? decision.policy_errors.length : 0;
  if (missing > 0) {
    return `${decisionText}, ${missing} missing`;
  }
  if (policy > 0) {
    return `${decisionText}, ${policy} policy issue${policy === 1 ? "" : "s"}`;
  }
  return decisionText;
}

function actionCaption(toolResults: Record<string, unknown>, actionKeys: string[]) {
  if (!actionKeys.length) {
    return "Skipped";
  }
  if ("refund" in toolResults) {
    return "Refund action";
  }
  if ("cancelled_order" in toolResults) {
    return "Cancel action";
  }
  if ("complaint" in toolResults) {
    return "Complaint logged";
  }
  if ("memory_write" in toolResults) {
    return "Memory written";
  }
  return "Action result";
}

function respondCaption(verifiedFacts: Record<string, unknown>) {
  const count = Object.keys(verifiedFacts).length;
  return count ? `${count} verified fact${count === 1 ? "" : "s"}` : "Response ready";
}

function stringValue(value: unknown) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  return String(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function isFlowNodeId(value: string): value is FlowNodeId {
  return value === "planner" || value === "read_tools" || value === "verifier" || value === "actions" || value === "respond";
}

function nodeTraceMaxStep(nodeTrace: StreamNodeEvent[]) {
  return nodeTrace.reduce((maxStep, event) => {
    if (!isFlowNodeId(event.node)) {
      return maxStep;
    }
    return Math.max(maxStep, nodeOrderIndex(event.node));
  }, -1);
}

function nodeOrderIndex(nodeId: FlowNodeId) {
  return ["planner", "read_tools", "verifier", "actions", "respond"].indexOf(nodeId);
}

function traceStatus(
  nodeId: FlowNodeId,
  traceByNode: Partial<Record<FlowNodeId, StreamNodeEvent>>,
  required: boolean,
): "active" | "skipped" {
  if (traceByNode[nodeId]) {
    return "active";
  }
  return required ? "skipped" : "skipped";
}

function combinedToolResults(nodeTrace: StreamNodeEvent[]) {
  return nodeTrace.reduce<Record<string, unknown>>((results, event) => {
    const toolResults = asRecord(event.state.tool_results);
    return { ...results, ...toolResults };
  }, {});
}

function summarizeValue(value: unknown) {
  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? "" : "s"}`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const status = stringValue(record.status);
    const orderId = stringValue(record.order_id);
    if (status && orderId) {
      return `order ${orderId}: ${status}`;
    }
    if (status) {
      return status;
    }
    return Object.keys(record).join(", ") || "object";
  }
  return stringValue(value) ?? "-";
}

function ResetDialog({
  isOpen,
  isResetting,
  onCancel,
  onConfirm,
}: {
  isOpen: boolean;
  isResetting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <div aria-modal="true" className="dialog" role="dialog">
        <h2>Reset demo data?</h2>
        <p>
          This will restore seeded orders, rebuild seeded complaints and memories, exit the
          current thread, and return to customer selection.
        </p>
        <div className="dialog-actions">
          <button className="ghost-button" disabled={isResetting} onClick={onCancel} type="button">
            Cancel
          </button>
          <button
            className="danger-button"
            disabled={isResetting}
            onClick={onConfirm}
            type="button"
          >
            {isResetting ? <Loader2 className="spin" size={16} /> : <RefreshCcw size={16} />}
            Reset and exit
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
