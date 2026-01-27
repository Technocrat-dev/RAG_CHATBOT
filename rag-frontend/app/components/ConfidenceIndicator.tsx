"use client";

interface ConfidenceIndicatorProps {
    confidence: number | null;
    iterations?: number;
    wasCorrected?: boolean;
}

export function ConfidenceIndicator({
    confidence,
    iterations = 1,
    wasCorrected = false
}: ConfidenceIndicatorProps) {
    if (confidence === null) return null;

    // Determine color based on confidence level
    const getColor = (conf: number) => {
        if (conf >= 0.8) return { bg: "bg-green-500", text: "text-green-400", label: "High" };
        if (conf >= 0.6) return { bg: "bg-yellow-500", text: "text-yellow-400", label: "Medium" };
        return { bg: "bg-red-500", text: "text-red-400", label: "Low" };
    };

    const { bg, text, label } = getColor(confidence);
    const percentage = Math.round(confidence * 100);

    return (
        <div className="flex items-center gap-3 mt-3 pt-3 border-t border-gray-700">
            {/* Confidence Bar */}
            <div className="flex-1">
                <div className="flex justify-between mb-1">
                    <span className="text-xs text-gray-400">Confidence</span>
                    <span className={`text-xs font-medium ${text}`}>{label} ({percentage}%)</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div
                        className={`h-1.5 rounded-full ${bg} transition-all duration-500`}
                        style={{ width: `${percentage}%` }}
                    />
                </div>
            </div>

            {/* Metadata */}
            <div className="flex gap-2 text-xs text-gray-500">
                {iterations > 1 && (
                    <span className="px-2 py-1 bg-gray-800 rounded" title="Retrieval iterations">
                        🔄 {iterations}x
                    </span>
                )}
                {wasCorrected && (
                    <span className="px-2 py-1 bg-purple-900/50 text-purple-400 rounded" title="Self-corrected">
                        ✨ Refined
                    </span>
                )}
            </div>
        </div>
    );
}
