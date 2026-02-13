import React, { ReactNode } from "react";
import {
    Breadcrumb,
    BreadcrumbItem as BreadcrumbItemUI,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { SearchForm } from "@/components/search-form"
import { LocationSelector } from "@/components/location-selector"
import Link from "next/link";

export interface BreadcrumbItem {
    label: string;
    href?: string;
}

interface PageProps {
    header: ReactNode;
    actions?: ReactNode;
    children?: ReactNode;
    footer?: ReactNode;
    breadcrumbs?: BreadcrumbItem[];
}

export function Page({
    header,
    actions,
    children,
    footer,
    breadcrumbs
}: PageProps) {
    return (
        <div className="flex flex-col min-h-svh animate-in fade-in duration-500">
            <header className="flex h-16 shrink-0 items-center justify-between gap-2 border-b px-4 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-10">
                <div className="flex items-center gap-2">
                    {breadcrumbs && breadcrumbs.length > 0 && (
                        <Breadcrumb>
                            <BreadcrumbList>
                                {breadcrumbs.map((item, index) => (
                                    <React.Fragment key={index}>
                                        <BreadcrumbItemUI>
                                            {item.href ? (
                                                <BreadcrumbLink asChild>
                                                    <Link href={item.href}>{item.label}</Link>
                                                </BreadcrumbLink>
                                            ) : (
                                                <BreadcrumbPage>{item.label}</BreadcrumbPage>
                                            )}
                                        </BreadcrumbItemUI>
                                        {index < breadcrumbs.length - 1 && (
                                            <BreadcrumbSeparator />
                                        )}
                                    </React.Fragment>
                                ))}
                            </BreadcrumbList>
                        </Breadcrumb>
                    )}
                </div>
                <div className="flex items-center gap-2 ml-auto">
                    <LocationSelector />
                    <SearchForm />
                </div>
            </header>

            <div className="flex-1 p-4 lg:p-6 space-y-8">
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
        </div>
    );
}
