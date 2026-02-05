"use client";

import { ReactNode } from "react";

interface PageShellProps {
    title: string;
    subtitle?: ReactNode;
    actions?: ReactNode;
    breadcrumbs?: ReactNode;
    children: ReactNode;
}

export function PageShell({ title, subtitle, actions, breadcrumbs, children }: PageShellProps) {
    return (
        <div className="flex flex-col space-y-8 animate-in fade-in duration-500">
            {breadcrumbs && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground -mb-4">
                    {breadcrumbs}
                </div>
            )}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="space-y-1.5">
                    <h1 className="text-4xl font-extrabold tracking-tight text-foreground bg-clip-text">
                        {title}
                    </h1>
                    {subtitle && (
                        <p className="text-lg text-muted-foreground font-medium">
                            {subtitle}
                        </p>
                    )}
                </div>
                {actions && (
                    <div className="flex items-center gap-3">
                        {actions}
                    </div>
                )}
            </div>
            <div className="flex-1">
                {children}
            </div>
        </div>
    );
}
