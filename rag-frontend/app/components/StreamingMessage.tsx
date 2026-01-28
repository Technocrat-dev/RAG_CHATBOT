"use client";
import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Loader2 } from "lucide-react";

interface StreamingMessageProps {
    content: string;
    isStreaming: boolean;
}

/**
 * Component that displays streaming text with a typing cursor effect.
 * Renders markdown as tokens arrive.
 */
export function StreamingMessage({ content, isStreaming }: StreamingMessageProps) {
    const [displayContent, setDisplayContent] = useState("");

    useEffect(() => {
        setDisplayContent(content);
    }, [content]);

    return (
        <div className="relative">
            <div className="prose prose-invert text-sm leading-relaxed break-words max-w-none">
                <ReactMarkdown>{displayContent}</ReactMarkdown>
            </div>

            {/* Streaming cursor */}
            {isStreaming && (
                <span className="inline-flex items-center ml-1">
                    <span className="w-2 h-4 bg-purple-500 animate-pulse rounded-sm" />
                </span>
            )}

            {/* Streaming indicator */}
            {isStreaming && (
                <div className="flex items-center gap-2 mt-2 text-xs text-purple-400">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    <span>Generating response...</span>
                </div>
            )}
        </div>
    );
}
