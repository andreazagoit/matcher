import Link from "next/link";
// Removed unused Card imports
import { Badge } from "@/components/ui/badge";
import { Users } from "lucide-react";

export interface SpaceCardProps {
    space: {
        id: string;
        name: string;
        slug: string;
        description?: string | null;
        visibility: string;
        isActive: boolean;
        membersCount: number;
        image?: string | null;
    };
}

export function SpaceCard({ space }: SpaceCardProps) {
    return (
        <Link href={`/spaces/${space.id}`} className="block h-full group">
            <div className="h-full flex flex-col gap-3">
                {/* Image Aspect Ratio */}
                <div className="aspect-square w-full relative bg-muted rounded-xl overflow-hidden ring-1 ring-border/50 group-hover:ring-primary/20 transition-all">
                    {space.image ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={space.image} alt={space.name} className="size-full object-cover transition-transform duration-500 group-hover:scale-105" />
                    ) : (
                        <div className="size-full flex items-center justify-center bg-muted/50">
                            <span className="text-6xl font-black text-muted-foreground/20 select-none group-hover:scale-110 transition-transform">
                                {space.name.charAt(0).toUpperCase()}
                            </span>
                        </div>
                    )}

                    {/* Overlay Badges */}
                    <div className="absolute top-2 right-2 flex gap-1">
                        {space.isActive && (
                            <Badge className="bg-white/90 text-black hover:bg-white shadow-sm backdrop-blur-sm h-6">
                                Active
                            </Badge>
                        )}
                    </div>
                </div>

                {/* Content Below */}
                <div className="space-y-1.5 px-1">
                    <div className="flex justify-between items-start gap-2">
                        <h3 className="font-semibold tracking-tight text-lg leading-tight group-hover:text-primary transition-colors">
                            {space.name}
                        </h3>
                    </div>

                    <p className="text-sm text-muted-foreground line-clamp-1">
                        {space.description || "No description"}
                    </p>

                    <div className="flex items-center gap-3 pt-1 text-xs text-muted-foreground">
                        <Badge variant="secondary" className="font-normal h-5 px-1.5 bg-muted/50">
                            {space.visibility === "public" ? "Public" : "Private"}
                        </Badge>
                        <div className="flex items-center gap-1.5">
                            <Users className="size-3.5" />
                            <span>{space.membersCount}</span>
                        </div>
                    </div>
                </div>
            </div>
        </Link>
    );
}
