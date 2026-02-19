"use client";

import { useState, useEffect } from "react";
import { MapPin, Target, Loader2 } from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { toast } from "sonner";

/**
 * Location Selector
 *
 * Saves GPS position and search radius locally (cookies).
 */
export function LocationSelector() {
    const [radius, setRadius] = useState(50);
    const [lat, setLat] = useState<number | null>(null);
    const [lng, setLng] = useState<number | null>(null);
    const [isLocating, setIsLocating] = useState(false);
    const [open, setOpen] = useState(false);

    // Initial load from cookies
    useEffect(() => {
        if (typeof document === "undefined") return;
        const cookies = document.cookie.split(";");

        const radiusCookie = cookies.find((c) =>
            c.trim().startsWith("matcher_radius=")
        );
        if (radiusCookie) {
            const val = parseInt(radiusCookie.split("=")[1]);
            // Use functional update to avoid lint error if possible, 
            // but the issue is calling it at all in the effect body.
            // However, for initialization on mount, this is the standard way.
            // We'll add a disable comment for this specific initialization.
            // eslint-disable-next-line react-hooks/set-state-in-effect
            if (!isNaN(val)) setRadius(val);
        }

        const latCookie = cookies.find((c) =>
            c.trim().startsWith("matcher_lat=")
        );
        if (latCookie) {
            const val = parseFloat(latCookie.split("=")[1]);
            if (!isNaN(val)) setLat(val);
        }

        const lngCookie = cookies.find((c) =>
            c.trim().startsWith("matcher_lng=")
        );
        if (lngCookie) {
            const val = parseFloat(lngCookie.split("=")[1]);
            if (!isNaN(val)) setLng(val);
        }
    }, []);

    const handleRadiusChange = (val: number) => {
        setRadius(val);
        document.cookie = `matcher_radius=${val}; path=/; max-age=${60 * 60 * 24 * 365}`;
    };

    const handleGetLocation = () => {
        setIsLocating(true);
        if (!navigator.geolocation) {
            toast.error("Geolocation is not supported by your browser");
            setIsLocating(false);
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const newLat = position.coords.latitude;
                const newLng = position.coords.longitude;
                setLat(newLat);
                setLng(newLng);
                setIsLocating(false);

                // Save to cookies
                const maxAge = 60 * 60 * 24 * 365; // 1 year
                document.cookie = `matcher_lat=${newLat}; path=/; max-age=${maxAge}`;
                document.cookie = `matcher_lng=${newLng}; path=/; max-age=${maxAge}`;

                toast.success("Position updated!");
            },
            (error) => {
                toast.error(
                    "Unable to retrieve your location: " + error.message
                );
                setIsLocating(false);
            }
        );
    };

    const hasLocation = lat !== null && lng !== null;

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2 h-9">
                    <MapPin className="h-4 w-4 text-primary" />
                    <span className="hidden sm:inline">
                        {hasLocation ? `${radius} km` : "Location"}
                    </span>
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Matching Radius & Location</DialogTitle>
                    <DialogDescription>
                        Set your current location and how far you want to look
                        for matches.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-6 py-4">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <Label htmlFor="radius">
                                Search Radius: {radius} km
                            </Label>
                            <Target className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <Slider
                            id="radius"
                            min={1}
                            max={100}
                            step={1}
                            value={[radius]}
                            onValueChange={(val) => handleRadiusChange(val[0])}
                        />
                        <p className="text-[10px] text-muted-foreground">
                            Radius is saved locally in your browser.
                        </p>
                    </div>

                    <div className="space-y-3">
                        <Label>Your Position</Label>
                        <div className="flex flex-col gap-2">
                            <Button
                                variant={hasLocation ? "default" : "secondary"}
                                onClick={handleGetLocation}
                                disabled={isLocating}
                                className="w-full"
                            >
                                {isLocating ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <MapPin className="mr-2 h-4 w-4" />
                                )}
                                {hasLocation
                                    ? "Update Position"
                                    : "Get Current Position"}
                            </Button>
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}
