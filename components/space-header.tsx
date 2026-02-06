import { Badge } from "@/components/ui/badge";
import { Users, Hash } from "lucide-react";
import { cn } from "@/lib/utils";

interface SpaceHeaderProps {
    space: {
        name: string;
        image?: string | null;
        description?: string | null;
        slug: string;
        membersCount?: number | null;
        visibility: string;
        createdAt?: string;
    };
    className?: string;
}

export function SpaceHeader({ space, className }: SpaceHeaderProps) {
    return (
        <div className={cn("flex flex-col md:flex-row gap-6 md:gap-8 items-start md:items-center", className)}>
            {/* Big Image Column */}
            <div className="shrink-0">
                <div className="h-48 w-48 md:h-64 md:w-64 rounded-2xl bg-muted border-2 border-border/50 overflow-hidden shadow-sm">
                    {space.image ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                            src={space.image}
                            alt={space.name}
                            className="h-full w-full object-cover"
                        />
                    ) : (
                        <div className="flex h-full w-full items-center justify-center bg-primary/5 text-primary/40">
                            <span className="text-5xl font-bold select-none">
                                {space.name.charAt(0).toUpperCase()}
                            </span>
                        </div>
                    )}
                </div>
            </div>

            {/* Info Column */}
            <div className="flex-1 space-y-4 min-w-0 py-1">

                {/* Title & Description Section */}
                <div className="space-y-2">
                    <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-foreground">
                        {space.name}
                    </h1>

                    {/* Bio / Description */}
                    {space.description && (
                        <p className="text-base md:text-lg text-muted-foreground leading-relaxed max-w-2xl font-medium">
                            {space.description}
                        </p>
                    )}
                </div>

                {/* Metadata Row */}
                <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground pt-2">
                    <Badge variant={space.visibility === "public" ? "secondary" : "outline"} className="capitalize">
                        {space.visibility}
                    </Badge>

                    <div className="flex items-center gap-1.5 px-2 py-0.5 bg-muted/50 rounded-md font-mono text-xs text-foreground/80">
                        <Hash className="h-3 w-3 opacity-70" />
                        {space.slug}
                    </div>

                    <div className="flex items-center gap-1.5">
                        <span className="text-border mx-1">•</span>
                        <Users className="h-4 w-4 opacity-70" />
                        <span>{space.membersCount || 0} members</span>
                    </div>
                </div>

            </div>
        </div>
    );
}
