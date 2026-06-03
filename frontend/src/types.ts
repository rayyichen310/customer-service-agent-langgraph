export type ChatRequest = {
  thread_id: string;
  message: string;
  customer_id: number | null;
};

export type ChatResponse = {
  thread_id: string;
  response: string;
  order_id: number | null;
  customer_id: number | null;
  tool_results: Record<string, unknown>;
  verified_facts: Record<string, unknown>;
  verification_decision: Record<string, unknown>;
};

export type StreamNodeEvent = {
  node: string;
  state: Record<string, unknown>;
};

export type ChatStreamEvent =
  | {
      event: "started";
      data: {
        thread_id: string;
        customer_id: number | null;
      };
    }
  | {
      event: "node";
      data: StreamNodeEvent;
    }
  | {
      event: "final";
      data: ChatResponse;
    }
  | {
      event: "error";
      data: {
        message: string;
      };
    };

export type CustomerCandidate = {
  id: number;
  name: string;
  email: string;
  orders: string;
};

export type CustomerSnapshot = {
  customer: Record<string, unknown> | null;
  orders: Array<Record<string, unknown>>;
  complaints: Array<Record<string, unknown>>;
  memories: Array<Record<string, unknown>>;
  issue_patterns: Record<string, unknown>;
};

export type MessageRecord = {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  createdAt: string;
  payload?: ChatResponse;
  nodeTrace?: StreamNodeEvent[];
};

export type ApiStatus = "checking" | "online" | "offline";
