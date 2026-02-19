"use client";

import { useState } from "react";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { MapPin, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command";
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

const RADIUS_OPTIONS = [
    { value: 10, label: "10 km" },
    { value: 25, label: "25 km" },
    { value: 50, label: "50 km" },
    { value: 100, label: "100 km" },
    { value: 250, label: "250 km" },
];

interface RadiusSelectorProps {
    value: number;
}

export function RadiusSelector({ value }: RadiusSelectorProps) {
    const [open, setOpen] = useState(false);
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();

    const handleSelect = (radius: number) => {
        const params = new URLSearchParams(searchParams.toString());
        params.set("radius", String(radius));
        router.push(`${pathname}?${params.toString()}`);
        setOpen(false);
    };

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button
                    variant="outline"
                    size="sm"
                    role="combobox"
                    aria-expanded={open}
                    className="gap-2 h-9 px-3 font-normal justify-between"
                >
                    <MapPin className="h-4 w-4 text-primary shrink-0" />
                    <span className="hidden sm:inline">{value} km</span>
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-48 p-0" align="end">
                <Command>
                    <CommandInput placeholder="Search radius..." />
                    <CommandList>
                        <CommandEmpty>No results.</CommandEmpty>
                        <CommandGroup heading="Radius">
                            {RADIUS_OPTIONS.map((opt) => (
                                <CommandItem
                                    key={opt.value}
                                    value={opt.label}
                                    onSelect={() => handleSelect(opt.value)}
                                >
                                    <Check
                                        className={cn(
                                            "mr-2 h-4 w-4",
                                            value === opt.value ? "opacity-100" : "opacity-0"
                                        )}
                                    />
                                    {opt.label}
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    );
}
