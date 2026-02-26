"use client";

import { useState } from "react";
import { MapPin, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMutation } from "@apollo/client/react";
import { UPDATE_LOCATION } from "@/lib/models/users/gql";
import type { UpdateLocationMutation } from "@/lib/graphql/__generated__/graphql";

export function LocationBanner() {
    const [dismissed, setDismissed] = useState(false);
    const [locating, setLocating] = useState(false);
    const [done, setDone] = useState(false);
    const [city, setCity] = useState<string | null>(null);

    const [updateLocation] = useMutation<UpdateLocationMutation>(UPDATE_LOCATION);

    if (dismissed || done) return null;

    const handleGetLocation = () => {
        if (!navigator.geolocation) return;
        setLocating(true);
        navigator.geolocation.getCurrentPosition(
            async (pos) => {
                try {
                    const res = await updateLocation({
                        variables: { lat: pos.coords.latitude, lon: pos.coords.longitude },
                    });
                    // Also persist to cookies so next SSR render picks up radius/lat/lon
                    const maxAge = 60 * 60 * 24 * 365;
                    document.cookie = `matcher_lat=${pos.coords.latitude}; path=/; max-age=${maxAge}`;
                    document.cookie = `matcher_lng=${pos.coords.longitude}; path=/; max-age=${maxAge}`;

                    setCity(res.data?.updateLocation?.location ?? null);
                    setDone(true);
                } catch {
                    // silently ignore — user can try again via header selector
                } finally {
                    setLocating(false);
                }
            },
            () => setLocating(false),
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 },
        );
    };

    return (
        <div className="relative flex items-center gap-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-5 py-4 text-sm">
            <MapPin className="h-5 w-5 shrink-0 text-amber-500" />
            <div className="flex-1">
                <p className="font-medium text-foreground">Posizione non impostata</p>
                <p className="text-muted-foreground text-xs mt-0.5">
                    Imposta la tua posizione per vedere i match vicini a te.
                </p>
            </div>
            <Button
                size="sm"
                variant="outline"
                className="shrink-0 border-amber-500/40 hover:bg-amber-500/10"
                onClick={handleGetLocation}
                disabled={locating}
            >
                {locating ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1.5" />
                ) : (
                    <MapPin className="h-4 w-4 mr-1.5" />
                )}
                {locating ? "Rilevamento…" : "Usa la mia posizione"}
            </Button>
            <button
                onClick={() => setDismissed(true)}
                className="absolute right-3 top-3 text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Chiudi"
            >
                <X className="h-4 w-4" />
            </button>
        </div>
    );
}
