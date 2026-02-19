"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Search, LayoutGrid, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useQuery } from "@apollo/client/react";
import { useSession } from "@/lib/auth-client";
import {
    Command,
    CommandDialog,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command";
import { GET_ALL_SPACES } from "@/lib/models/spaces/gql";
import type { GetAllSpacesQuery } from "@/lib/graphql/__generated__/graphql";

export function SearchForm() {
    const [open, setOpen] = React.useState(false);
    const router = useRouter();
    const { data: session } = useSession();

    const { data } = useQuery<GetAllSpacesQuery>(GET_ALL_SPACES, { skip: !open });
    const spaces = data?.spaces ?? [];

    // ⌘K shortcut
    React.useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "k") {
                e.preventDefault();
                setOpen((o) => !o);
            }
        };
        document.addEventListener("keydown", handler);
        return () => document.removeEventListener("keydown", handler);
    }, []);

    const navigate = (href: string) => {
        router.push(href);
        setOpen(false);
    };

    return (
        <>
            <Button variant="outline" size="sm" onClick={() => setOpen(true)} className="gap-2 h-9">
                <Search className="h-4 w-4 text-primary" />
                <span className="hidden sm:inline">Search</span>
            </Button>

            <CommandDialog open={open} onOpenChange={setOpen}>
                <Command>
                <CommandInput placeholder="Search spaces, people..." />
                <CommandList>
                    <CommandEmpty>No results found.</CommandEmpty>

                    {spaces.length > 0 && (
                        <CommandGroup heading="Spaces">
                            {spaces.map((space) => (
                                <CommandItem
                                    key={space.id}
                                    value={space.name}
                                    onSelect={() => navigate(`/spaces/${space.slug}`)}
                                >
                                    <LayoutGrid className="mr-2 h-4 w-4 text-muted-foreground" />
                                    {space.name}
                                    {space.description && (
                                        <span className="ml-2 text-xs text-muted-foreground truncate">
                                            {space.description}
                                        </span>
                                    )}
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    )}

                    <CommandGroup heading="Navigation">
                        <CommandItem onSelect={() => navigate("/")}>
                            <Search className="mr-2 h-4 w-4 text-muted-foreground" />
                            Feed
                        </CommandItem>
                        <CommandItem onSelect={() => navigate("/discover")}>
                            <Search className="mr-2 h-4 w-4 text-muted-foreground" />
                            Discover
                        </CommandItem>
                        <CommandItem onSelect={() => navigate("/messages")}>
                            <Search className="mr-2 h-4 w-4 text-muted-foreground" />
                            Messages
                        </CommandItem>
                        {(() => { const u = (session?.user as Record<string, unknown>)?.username as string | undefined; return u ? (
                            <CommandItem onSelect={() => navigate(`/users/${u}`)}>
                                <User className="mr-2 h-4 w-4 text-muted-foreground" />
                                My Profile
                            </CommandItem>
                        ) : null; })()}
                    </CommandGroup>
                </CommandList>
                </Command>
            </CommandDialog>
        </>
    );
}
