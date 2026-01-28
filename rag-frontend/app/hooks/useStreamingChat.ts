"use client";
import { useState, useCallback } from "react";

interface LogEvent {
    step: string;
    message: string;
    warning?: boolean;
}

interface StreamingOptions {
    onLog?: (log: LogEvent) => void;
    onToken?: (token: string) => void;
    onComplete?: (data: { confidence: number; sources: string[]; wasCorrected: boolean }) => void;
    onError?: (error: string) => void;
}

interface StreamEvent {
    type: "log" | "token" | "done" | "error";
    step?: string;
    message?: string;
    content?: string;
    warning?: boolean;
    confidence?: number;
    sources?: string[];
    was_corrected?: boolean;
}

/**
 * Hook for streaming chat responses with processing logs.
 * Handles new event types: log, token, done, error
 */
export function useStreamingChat() {
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamedContent, setStreamedContent] = useState("");
    const [logs, setLogs] = useState<LogEvent[]>([]);

    const streamChat = useCallback(async (
        query: string,
        sessionId: string = "default",
        options: StreamingOptions = {},
        collectionId: string = "default"
    ) => {
        setIsStreaming(true);
        setStreamedContent("");
        setLogs([]);
        let fullContent = "";

        try {
            const response = await fetch("/api/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    model: "llama3",
                    session_id: sessionId,
                    collection_id: collectionId,
                    use_self_correction: true
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader = response.body?.getReader();
            if (!reader) throw new Error("No response body");

            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split("\n").filter(line => line.trim());

                for (const line of lines) {
                    try {
                        const event: StreamEvent = JSON.parse(line);

                        switch (event.type) {
                            case "log":
                                const logEntry: LogEvent = {
                                    step: event.step || "",
                                    message: event.message || "",
                                    warning: event.warning
                                };
                                setLogs(prev => [...prev, logEntry]);
                                options.onLog?.(logEntry);
                                break;

                            case "token":
                                if (event.content) {
                                    fullContent += event.content;
                                    setStreamedContent(fullContent);
                                    options.onToken?.(event.content);
                                }
                                break;

                            case "done":
                                options.onComplete?.({
                                    confidence: event.confidence || 0,
                                    sources: event.sources || [],
                                    wasCorrected: event.was_corrected || false
                                });
                                break;

                            case "error":
                                options.onError?.(event.message || "Unknown error");
                                break;
                        }
                    } catch {
                        // Skip non-JSON lines
                    }
                }
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : "Stream failed";
            options.onError?.(message);
        } finally {
            setIsStreaming(false);
        }

        return fullContent;
    }, []);

    const reset = useCallback(() => {
        setStreamedContent("");
        setLogs([]);
        setIsStreaming(false);
    }, []);

    return {
        streamChat,
        isStreaming,
        streamedContent,
        logs,
        reset
    };
}
