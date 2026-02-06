import { ReactNode } from "react";
// removed unused import;

interface PageShellProps {
    header: ReactNode;
    actions?: ReactNode;
    children?: ReactNode;
    footer?: ReactNode;
}

export function PageShell({
    header,
    actions,
    children,
    footer
}: PageShellProps) {
    return (
        <div className="flex flex-col space-y-8 animate-in fade-in duration-500">
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                    {header}
                </div>
                {actions && (
                    <div className="shrink-0">
                        {actions}
                    </div>
                )}
            </div>

            {children && (
                <div className="flex-1">
                    {children}
                </div>
            )}

            {footer && (
                <div className="mt-auto pt-8">
                    {footer}
                </div>
            )}
        </div>
    );
}
