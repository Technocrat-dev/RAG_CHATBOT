"use client";
import { useState, useRef, useEffect, ChangeEvent, FormEvent } from "react";
import { Send, Upload, Loader2, Bot, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { ConfidenceIndicator } from "./components/ConfidenceIndicator";
import { SourcesPanel } from "./components/SourcesPanel";

// Message type with enhanced metadata
type Message = {
  role: "system" | "user" | "assistant";
  content: string;
  confidence?: number | null;
  sources?: string[];
  iterations?: number;
  wasCorrected?: boolean;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hello! Upload any document (PDF, image) and I'll help you understand it. I use **self-correcting AI** to ensure accurate answers." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [useSelfCorrection, setUseSelfCorrection] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // --- HANDLE FILE UPLOAD ---
  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `✅ **${file.name}** indexed successfully!\n\n📄 Type: ${data.handler || 'Document'}\n📦 Chunks: ${data.message}`
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "system", content: "❌ Upload failed. Please try again." },
      ]);
    } finally {
      setUploading(false);
    }
  };

  // --- HANDLE CHAT ---
  const handleChat = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input;
    setInput("");

    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMessage,
          model: "llama3",
          use_self_correction: useSelfCorrection
        }),
      });
      const data = await res.json();

      setMessages((prev) => [
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
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "system", content: "❌ Error connecting to server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100 font-sans">
      {/* HEADER */}
      <header className="flex items-center justify-between p-4 border-b border-gray-800 bg-gray-950">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            Self-Correcting RAG
          </h1>
          {/* Self-Correction Toggle */}
          <button
            onClick={() => setUseSelfCorrection(!useSelfCorrection)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition ${useSelfCorrection
                ? "bg-purple-600/20 text-purple-400 border border-purple-500/30"
                : "bg-gray-800 text-gray-400 border border-gray-700"
              }`}
          >
            <Sparkles className="w-3 h-3" />
            {useSelfCorrection ? "Self-Correct ON" : "Self-Correct OFF"}
          </button>
        </div>
        <div className="relative">
          <input
            type="file"
            onChange={handleUpload}
            accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.txt,.md"
            className="absolute inset-0 opacity-0 cursor-pointer"
            disabled={uploading}
          />
          <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition disabled:opacity-50">
            {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            {uploading ? "Indexing..." : "Upload"}
          </button>
        </div>
      </header>

      {/* CHAT AREA */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] p-4 rounded-2xl ${msg.role === "user"
                ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-br-none"
                : msg.role === "system"
                  ? "bg-gray-800/50 text-green-400 text-sm border border-green-900/50"
                  : "bg-gray-800 text-gray-100 rounded-bl-none border border-gray-700"
              }`}>
              {msg.role === "assistant" && (
                <div className="flex items-center gap-2 mb-2 text-xs text-gray-400 uppercase tracking-wider font-bold">
                  <Bot className="w-3 h-3" /> AI Assistant
                </div>
              )}
              <div className="prose prose-invert text-sm leading-relaxed break-words max-w-none">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>

              {/* Confidence & Sources for assistant messages */}
              {msg.role === "assistant" && (
                <>
                  <ConfidenceIndicator
                    confidence={msg.confidence ?? null}
                    iterations={msg.iterations}
                    wasCorrected={msg.wasCorrected}
                  />
                  <SourcesPanel sources={msg.sources || []} />
                </>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 p-4 rounded-2xl rounded-bl-none flex items-center gap-3 text-gray-400 border border-gray-700">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <span className="text-sm">
                {useSelfCorrection ? "Analyzing & validating..." : "Thinking..."}
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* INPUT AREA */}
      <form onSubmit={handleChat} className="p-4 bg-gray-950 border-t border-gray-800">
        <div className="flex gap-2 max-w-4xl mx-auto">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none placeholder-gray-500 transition"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-xl disabled:opacity-50 transition"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
}