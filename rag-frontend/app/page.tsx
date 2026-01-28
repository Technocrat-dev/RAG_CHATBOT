"use client";
import { useState, useRef, useEffect, ChangeEvent, FormEvent } from "react";
import { Send, Upload, Loader2, Bot, Sparkles, Zap, Plus, MessageSquare, Trash2, FileText } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { ConfidenceIndicator } from "./components/ConfidenceIndicator";
import { SourcesPanel } from "./components/SourcesPanel";
import { StreamingMessage } from "./components/StreamingMessage";
import { ProcessingLogs } from "./components/ProcessingLogs";
import { GlassCard } from "./components/GlassCard";
import { useStreamingChat } from "./hooks/useStreamingChat";

// Types
type Message = {
  role: "system" | "user" | "assistant";
  content: string;
  confidence?: number | null;
  sources?: string[];
  iterations?: number;
  wasCorrected?: boolean;
  isStreaming?: boolean;
};

type ChatSession = {
  id: string;
  name: string;
  collection_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export default function Home() {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hello! Click **New Chat** to start, then upload a document for that chat thread." },
  ]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [showLogs, setShowLogs] = useState(true);

  // Toggle states
  const [useSelfCorrection, setUseSelfCorrection] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);

  // Session management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentCollectionId, setCurrentCollectionId] = useState<string>("default");
  const [loadingSessions, setLoadingSessions] = useState(false);

  // Streaming hook
  const { streamChat, isStreaming, logs, reset } = useStreamingChat();
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    setLoadingSessions(true);
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (e) {
      console.error("Failed to load sessions:", e);
    } finally {
      setLoadingSessions(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Create new chat session with unique collection
  const handleNewChat = async () => {
    try {
      // Create a unique collection for this chat
      const collectionRes = await fetch("/api/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "Chat Documents", description: "Documents for this chat" })
      });
      const collectionData = await collectionRes.json();
      const collectionId = collectionData.collection?.id || `chat-${Date.now()}`;

      // Create session linked to this collection
      const res = await fetch(`/api/sessions?name=New Chat&collection_id=${collectionId}`, { method: "POST" });
      const data = await res.json();

      if (data.session) {
        setSessions(prev => [data.session, ...prev]);
        setCurrentSessionId(data.session.id);
        setCurrentCollectionId(collectionId);
        setMessages([
          { role: "assistant", content: "New chat started! Upload a document to ask questions about it. Documents in this chat are isolated from other chats." }
        ]);
        reset();
      }
    } catch (e) {
      console.error("Failed to create session:", e);
    }
  };

  // Load a session's messages
  const handleSelectSession = async (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setCurrentCollectionId(session.collection_id || "default");

    try {
      const res = await fetch(`/api/sessions/${session.id}`);
      const data = await res.json();

      if (data.messages && data.messages.length > 0) {
        const loadedMessages: Message[] = data.messages.map((m: any) => ({
          role: m.role,
          content: m.content,
          confidence: m.confidence,
          sources: m.sources
        }));
        setMessages(loadedMessages);
      } else {
        setMessages([
          { role: "assistant", content: "Chat loaded. Ask me anything about your documents!" }
        ]);
      }
      reset();
    } catch (e) {
      console.error("Failed to load session:", e);
    }
  };

  // Delete a session
  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (!confirm("Delete this chat?")) return;

    try {
      await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
      setSessions(prev => prev.filter(s => s.id !== sessionId));

      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setCurrentCollectionId("default");
        setMessages([
          { role: "assistant", content: "Hello! Click **New Chat** to start a conversation." }
        ]);
      }
    } catch (e) {
      console.error("Failed to delete session:", e);
    }
  };

  // Save message to backend
  const saveMessage = async (role: string, content: string, confidence?: number, sources?: string[]) => {
    if (!currentSessionId) return;

    try {
      await fetch(`/api/sessions/${currentSessionId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role, content, confidence, sources })
      });
    } catch (e) {
      console.error("Failed to save message:", e);
    }
  };

  // Update session name based on first message
  const updateSessionName = async (name: string) => {
    if (!currentSessionId) return;

    // Truncate to 30 chars
    const truncatedName = name.length > 30 ? name.substring(0, 30) + "..." : name;

    try {
      await fetch(`/api/sessions/${currentSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: truncatedName })
      });

      setSessions(prev => prev.map(s =>
        s.id === currentSessionId ? { ...s, name: truncatedName } : s
      ));
    } catch (e) {
      console.error("Failed to update session name:", e);
    }
  };

  // Handle file upload - uploads to current collection
  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!currentSessionId) {
      alert("Please create a New Chat first, then upload a document to it.");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Upload directly to backend (bypass Next.js proxy for large file uploads)
      const res = await fetch(`http://127.0.0.1:8000/upload?collection_id=${currentCollectionId}`, {
        method: "POST",
        body: formData,
      });

      // Check if response is OK before parsing JSON
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();

      const systemMsg = `✅ **${file.name}** indexed to this chat! ${data.message}`;
      setMessages(prev => [...prev, { role: "system", content: systemMsg }]);

      // Update session name to document name if it's still "New Chat"
      const currentSession = sessions.find(s => s.id === currentSessionId);
      if (currentSession?.name === "New Chat") {
        updateSessionName(file.name.replace(/\.[^/.]+$/, "")); // Remove extension
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Upload failed";
      setMessages(prev => [...prev, { role: "system", content: `❌ ${errorMsg}` }]);
    } finally {
      setUploading(false);
    }
  };

  // Handle chat submission
  const handleChat = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    if (!currentSessionId) {
      alert("Please create a New Chat first.");
      return;
    }

    const userMessage = input;
    setInput("");
    reset();

    setMessages(prev => [...prev, { role: "user", content: userMessage }]);

    // Save user message to backend
    saveMessage("user", userMessage);

    // Note: Session name is updated when first document is uploaded (see handleUpload)

    if (useStreaming) {
      setMessages(prev => [...prev, { role: "assistant", content: "", isStreaming: true }]);

      let finalContent = "";
      let finalConfidence = 0;
      let finalSources: string[] = [];

      await streamChat(userMessage, currentSessionId, {
        onToken: (token) => {
          finalContent += token;
          setMessages(prev => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (updated[lastIdx]?.isStreaming) {
              updated[lastIdx] = { ...updated[lastIdx], content: finalContent };
            }
            return updated;
          });
        },
        onComplete: (data) => {
          finalConfidence = data.confidence;
          finalSources = data.sources;

          setMessages(prev => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (updated[lastIdx]) {
              updated[lastIdx] = {
                ...updated[lastIdx],
                isStreaming: false,
                confidence: data.confidence,
                sources: data.sources,
                wasCorrected: data.wasCorrected
              };
            }
            return updated;
          });

          // Save assistant response to backend
          saveMessage("assistant", finalContent, data.confidence, data.sources);
        },
        onError: (error) => {
          setMessages(prev => [...prev, { role: "system", content: `❌ Error: ${error}` }]);
        }
      }, currentCollectionId);
    } else {
      // Non-streaming mode
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: userMessage,
            model: "llama3",
            use_self_correction: useSelfCorrection,
            session_id: currentSessionId,
            collection_id: currentCollectionId
          }),
        });
        const data = await res.json();

        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            content: data.response,
            confidence: data.confidence,
            sources: data.context_used,
            iterations: data.iterations,
            wasCorrected: data.was_corrected
          },
        ]);

        // Save assistant response
        saveMessage("assistant", data.response, data.confidence, data.context_used);
      } catch {
        setMessages(prev => [...prev, { role: "system", content: "❌ Error connecting to server." }]);
      }
    }

    // Refresh sessions to update message count
    loadSessions();
  };

  return (
    <div className="flex h-screen" style={{ background: "var(--bg-deep)" }}>
      {/* Sidebar */}
      <aside className="w-64 border-r flex flex-col" style={{
        borderColor: "var(--border-subtle)",
        background: "var(--bg-base)"
      }}>
        {/* Logo */}
        <div className="p-4 border-b" style={{ borderColor: "var(--border-subtle)" }}>
          <h1 className="text-xl font-bold gradient-text">⚡ NeuralRAG</h1>
          <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            Self-Correcting AI
          </p>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl btn-primary text-sm font-medium text-white"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* Recent Chats */}
        <div className="flex-1 px-3 py-2 overflow-y-auto">
          <p className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>
            Your Chats
          </p>
          <div className="space-y-1">
            {loadingSessions ? (
              <div className="flex items-center gap-2 px-3 py-2 text-xs" style={{ color: "var(--text-muted)" }}>
                <Loader2 className="w-3 h-3 animate-spin" /> Loading...
              </div>
            ) : sessions.length === 0 ? (
              <p className="px-3 py-2 text-xs" style={{ color: "var(--text-muted)" }}>
                No chats yet. Click "New Chat" to start!
              </p>
            ) : (
              sessions.map(session => (
                <div
                  key={session.id}
                  onClick={() => handleSelectSession(session)}
                  className={`group flex items-center gap-2 px-3 py-2 rounded-lg text-sm cursor-pointer transition ${currentSessionId === session.id
                    ? "bg-cyan-500/10 border border-cyan-500/30"
                    : "hover:bg-white/5"
                    }`}
                  style={{ color: "var(--text-secondary)" }}
                >
                  <FileText className="w-4 h-4 flex-shrink-0" style={{ color: "var(--primary-dim)" }} />
                  <div className="flex-1 min-w-0">
                    <p className="truncate font-medium">{session.name}</p>
                    <p className="text-xs truncate" style={{ color: "var(--text-muted)" }}>
                      {session.message_count} messages
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(session.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition"
                    title="Delete chat"
                  >
                    <Trash2 className="w-3 h-3 text-red-400" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Current Collection Info */}
        {currentSessionId && (
          <div className="p-3 border-t text-xs" style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}>
            <p>Documents in this chat are isolated.</p>
          </div>
        )}
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 border-b" style={{
          borderColor: "var(--border-subtle)",
          background: "var(--bg-surface)"
        }}>
          <div className="flex items-center gap-3">
            <h2 className="font-semibold" style={{ color: "var(--text-primary)" }}>
              {sessions.find(s => s.id === currentSessionId)?.name || "NeuralRAG"}
            </h2>

            {/* Self-Correction Toggle */}
            <button
              onClick={() => setUseSelfCorrection(!useSelfCorrection)}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition"
              style={{
                background: useSelfCorrection ? "rgba(128, 0, 255, 0.15)" : "var(--bg-elevated)",
                color: useSelfCorrection ? "#A855F7" : "var(--text-secondary)",
                border: `1px solid ${useSelfCorrection ? "rgba(128, 0, 255, 0.3)" : "var(--border-subtle)"}`
              }}
            >
              <Sparkles className="w-3 h-3" />
              {useSelfCorrection ? "Self-Correct ON" : "Self-Correct OFF"}
            </button>

            {/* Streaming Toggle */}
            <button
              onClick={() => setUseStreaming(!useStreaming)}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition"
              style={{
                background: useStreaming ? "rgba(0, 255, 136, 0.15)" : "var(--bg-elevated)",
                color: useStreaming ? "var(--success)" : "var(--text-secondary)",
                border: `1px solid ${useStreaming ? "rgba(0, 255, 136, 0.3)" : "var(--border-subtle)"}`
              }}
            >
              <Zap className="w-3 h-3" />
              {useStreaming ? "Streaming ON" : "Streaming OFF"}
            </button>

            {/* Logs Toggle */}
            {useStreaming && (
              <button
                onClick={() => setShowLogs(!showLogs)}
                className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition"
                style={{
                  background: showLogs ? "rgba(0, 217, 255, 0.1)" : "var(--bg-elevated)",
                  color: showLogs ? "var(--primary)" : "var(--text-secondary)",
                  border: `1px solid ${showLogs ? "var(--border-bright)" : "var(--border-subtle)"}`
                }}
              >
                {showLogs ? "Logs ON" : "Logs OFF"}
              </button>
            )}
          </div>

          {/* Upload Button */}
          <div className="relative">
            <input
              type="file"
              onChange={handleUpload}
              accept=".pdf,.png,.jpg,.jpeg,.txt,.md"
              className="absolute inset-0 opacity-0 cursor-pointer"
              disabled={uploading || !currentSessionId}
            />
            <button
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition ${!currentSessionId ? 'opacity-50' : ''}`}
              style={{
                background: "var(--bg-elevated)",
                color: "var(--text-primary)",
                border: "1px solid var(--border-default)"
              }}
              title={currentSessionId ? "Upload to this chat" : "Create a New Chat first"}
            >
              {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {uploading ? "Indexing..." : "Upload"}
            </button>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Processing Logs */}
          {showLogs && useStreaming && (logs.length > 0 || isStreaming) && (
            <ProcessingLogs logs={logs} isProcessing={isStreaming && logs.length > 0} />
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <GlassCard
                className={`max-w-[75%] p-4 ${msg.role === "user" ? "rounded-br-sm" : "rounded-bl-sm"}`}
                elevated={msg.role === "assistant"}
                glow={msg.role === "assistant" && !msg.isStreaming}
              >
                {msg.role === "assistant" && (
                  <div className="flex items-center gap-2 mb-2 text-xs uppercase tracking-wider font-medium" style={{ color: "var(--primary)" }}>
                    <Bot className="w-3 h-3" />
                    AI Assistant
                    {msg.isStreaming && (
                      <span className="flex items-center gap-1 normal-case font-normal" style={{ color: "var(--success)" }}>
                        <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--success)" }} />
                        streaming
                      </span>
                    )}
                    {msg.wasCorrected && !msg.isStreaming && (
                      <span className="px-2 py-0.5 rounded-full text-[10px]" style={{
                        background: "rgba(0, 102, 255, 0.2)",
                        color: "var(--accent)"
                      }}>
                        ✨ Refined
                      </span>
                    )}
                  </div>
                )}

                {msg.role === "system" ? (
                  <div className="text-sm" style={{ color: "var(--success)" }}>
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : msg.isStreaming ? (
                  <StreamingMessage content={msg.content} isStreaming={true} />
                ) : (
                  <div className="prose prose-invert text-sm leading-relaxed max-w-none" style={{ color: "var(--text-primary)" }}>
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                )}

                {msg.role === "assistant" && !msg.isStreaming && msg.confidence !== undefined && (
                  <>
                    <ConfidenceIndicator
                      confidence={msg.confidence ?? null}
                      iterations={msg.iterations}
                      wasCorrected={msg.wasCorrected}
                    />
                    <SourcesPanel sources={msg.sources || []} />
                  </>
                )}
              </GlassCard>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <form onSubmit={handleChat} className="p-4 border-t" style={{
          borderColor: "var(--border-subtle)",
          background: "var(--bg-surface)"
        }}>
          <div className="flex gap-3 max-w-4xl mx-auto">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={currentSessionId ? "Ask about your documents..." : "Create a New Chat to start..."}
              className="flex-1 px-4 py-3 rounded-xl text-sm transition"
              style={{
                background: "var(--bg-elevated)",
                color: "var(--text-primary)",
                border: "1px solid var(--border-default)"
              }}
              disabled={isStreaming || !currentSessionId}
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim() || !currentSessionId}
              className="px-4 py-3 rounded-xl btn-primary disabled:opacity-50 transition"
            >
              <Send className="w-5 h-5 text-white" />
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}