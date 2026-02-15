"use client";

/**
 * Auth provider wrapper.
 * better-auth uses cookie-based sessions — no client-side provider needed.
 * This component is kept as a passthrough for layout compatibility.
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
    return <>{children}</>;
}
