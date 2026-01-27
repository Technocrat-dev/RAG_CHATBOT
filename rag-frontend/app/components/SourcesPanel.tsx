"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp, FileText } from "lucide-react";

interface SourcesPanelProps {
    sources: string[];
    isOpen?: boolean;
}

export function SourcesPanel({ sources, isOpen = false }: SourcesPanelProps) {
    const [expanded, setExpanded] = useState(isOpen);

    if (!sources || sources.length === 0) return null;

    return (
        <div className="mt-3 pt-3 border-t border-gray-700">
            {/* Toggle Header */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-2 w-full text-left text-xs text-gray-400 hover:text-gray-300 transition"
            >
                <FileText className="w-3 h-3" />
                <span className="font-medium">Sources ({sources.length})</span>
                {expanded ? <ChevronUp className="w-3 h-3 ml-auto" /> : <ChevronDown className="w-3 h-3 ml-auto" />}
            </button>

            {/* Sources List */}
            {expanded && (
                <div className="mt-2 space-y-2 max-h-60 overflow-y-auto">
                    {sources.map((source, idx) => (
                        <div
                            key={idx}
                            className="p-2 bg-gray-900/50 rounded-lg border border-gray-700/50 text-xs"
                        >
                            <div className="flex items-center gap-2 mb-1 text-gray-500">
                                <span className="px-1.5 py-0.5 bg-blue-900/50 text-blue-400 rounded text-[10px] font-medium">
                                    #{idx + 1}
                                </span>
                            </div>
                            <p className="text-gray-300 line-clamp-3 leading-relaxed">
                                {source.length > 200 ? source.substring(0, 200) + "..." : source}
                            </p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
