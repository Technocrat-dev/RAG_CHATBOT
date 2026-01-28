"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp, Terminal, CheckCircle, AlertTriangle, Loader2 } from "lucide-react";

interface LogEvent {
    step: string;
    message: string;
    warning?: boolean;
}

interface ProcessingLogsProps {
    logs: LogEvent[];
    isProcessing: boolean;
}

/**
 * Collapsible panel showing real-time processing steps.
 * Shows retrieval, re-ranking, validation steps etc.
 */
export function ProcessingLogs({ logs, isProcessing }: ProcessingLogsProps) {
    const [expanded, setExpanded] = useState(true);

    if (logs.length === 0 && !isProcessing) return null;

    // Get step icon
    const getStepIcon = (step: string, warning?: boolean) => {
        if (warning) return <AlertTriangle className="w-3 h-3 text-yellow-400" />;

        switch (step) {
            case "start":
            case "iteration":
                return <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />;
            case "complete":
                return <CheckCircle className="w-3 h-3 text-green-400" />;
            default:
                return <CheckCircle className="w-3 h-3 text-cyan-400" />;
        }
    };

    return (
        <div className="mb-4 rounded-xl border border-cyan-500/20 bg-slate-900/50 backdrop-blur-sm overflow-hidden">
            {/* Header */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between px-4 py-2 bg-slate-800/50 hover:bg-slate-800/70 transition"
            >
                <div className="flex items-center gap-2 text-xs text-cyan-400 font-medium">
                    <Terminal className="w-4 h-4" />
                    <span>Processing Steps</span>
                    {isProcessing && (
                        <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
                    )}
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">{logs.length} steps</span>
                    {expanded ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                    ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                    )}
                </div>
            </button>

            {/* Log entries */}
            {expanded && (
                <div className="px-4 py-3 space-y-2 max-h-48 overflow-y-auto">
                    {logs.map((log, idx) => (
                        <div
                            key={idx}
                            className={`flex items-start gap-2 text-xs ${log.warning ? "text-yellow-400" : "text-gray-400"
                                }`}
                        >
                            <div className="mt-0.5">
                                {getStepIcon(log.step, log.warning)}
                            </div>
                            <span>{log.message}</span>
                        </div>
                    ))}

                    {isProcessing && logs.length > 0 && (
                        <div className="flex items-center gap-2 text-xs text-cyan-400">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            <span>Processing...</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
