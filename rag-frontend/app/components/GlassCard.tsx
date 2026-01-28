"use client";
import { ReactNode } from "react";

interface GlassCardProps {
    children: ReactNode;
    className?: string;
    elevated?: boolean;
    glow?: boolean;
    hover?: boolean;
}

/**
 * Glassmorphism card component with optional glow and hover effects.
 */
export function GlassCard({
    children,
    className = "",
    elevated = false,
    glow = false,
    hover = false
}: GlassCardProps) {
    const baseClasses = elevated ? "glass-elevated" : "glass";
    const glowClass = glow ? "glow-border" : "";
    const hoverClass = hover ? "card-hover" : "";

    return (
        <div className={`rounded-2xl ${baseClasses} ${glowClass} ${hoverClass} ${className}`}>
            {children}
        </div>
    );
}
