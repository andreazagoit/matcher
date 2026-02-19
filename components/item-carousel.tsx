"use client";

import { useRef, useState, useEffect, useCallback, Children } from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ItemCarouselProps {
    children: React.ReactNode;
    title?: string;
    titleHref?: string;
    columns?: number; // visible items per page, default 4
    gap?: number;     // px, default 16
    className?: string;
}

export function ItemCarousel({
    children,
    title,
    titleHref,
    columns = 4,
    gap = 16,
    className,
}: ItemCarouselProps) {
    const wrapperRef = useRef<HTMLDivElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const [itemWidth, setItemWidth] = useState(0);
    const [canScrollLeft, setCanScrollLeft] = useState(false);
    const [canScrollRight, setCanScrollRight] = useState(false);

    const computeItemWidth = useCallback(() => {
        const el = wrapperRef.current;
        if (!el) return;
        const totalGap = gap * (columns - 1);
        setItemWidth((el.clientWidth - totalGap) / columns);
    }, [columns, gap]);

    const updateArrows = useCallback(() => {
        const el = scrollRef.current;
        if (!el) return;
        setCanScrollLeft(el.scrollLeft > 4);
        setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
    }, []);

    useEffect(() => {
        computeItemWidth();
        updateArrows();

        const ro = new ResizeObserver(() => {
            computeItemWidth();
            updateArrows();
        });
        if (wrapperRef.current) ro.observe(wrapperRef.current);

        const scrollEl = scrollRef.current;
        scrollEl?.addEventListener("scroll", updateArrows, { passive: true });

        return () => {
            ro.disconnect();
            scrollEl?.removeEventListener("scroll", updateArrows);
        };
    }, [computeItemWidth, updateArrows]);

    const scroll = (dir: "left" | "right") => {
        const el = scrollRef.current;
        if (!el) return;
        const pageWidth = (itemWidth + gap) * columns;
        el.scrollBy({ left: dir === "left" ? -pageWidth : pageWidth, behavior: "smooth" });
    };

    const items = Children.toArray(children);

    return (
        <div ref={wrapperRef} className={cn("space-y-4", className)}>
            {/* Header row: title on left, arrows on right */}
            {title && (
                <div className="flex items-center justify-between">
                    {titleHref ? (
                        <Link href={titleHref} className="group flex items-center gap-1.5 hover:underline underline-offset-4">
                            <h2 className="text-xl font-bold tracking-tight">{title}</h2>
                            <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                        </Link>
                    ) : (
                        <h2 className="text-xl font-bold tracking-tight">{title}</h2>
                    )}
                    <div className="flex items-center gap-1">
                        <Button
                            variant="outline"
                            size="icon"
                            onClick={() => scroll("left")}
                            disabled={!canScrollLeft}
                            className="h-8 w-8 rounded-full"
                        >
                            <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <Button
                            variant="outline"
                            size="icon"
                            onClick={() => scroll("right")}
                            disabled={!canScrollRight}
                            className="h-8 w-8 rounded-full"
                        >
                            <ChevronRight className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            )}

            {/* Scroll container */}
            <div
                ref={scrollRef}
                className="flex overflow-x-auto scroll-smooth no-scrollbar"
                style={{ gap }}
            >
                {itemWidth > 0 && items.map((child, i) => (
                    <div key={i} className="shrink-0" style={{ width: itemWidth }}>
                        {child}
                    </div>
                ))}
            </div>
        </div>
    );
}
