"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery } from "@apollo/client/react";
import { MapPin, Target, Save, Loader2 } from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { UPDATE_USER, GET_ME } from "@/lib/models/users/gql";
import { toast } from "sonner";
import type { GetMeQuery, UpdateUserMutation, UpdateUserMutationVariables } from "@/lib/graphql/__generated__/graphql";

export function LocationSelector() {
    const [open, setOpen] = useState(false);
    const [radius, setRadius] = useState(50);
    const [lat, setLat] = useState<number | null>(null);
    const [lng, setLng] = useState<number | null>(null);
    const [isLocating, setIsLocating] = useState(false);

    const { data: meData } = useQuery<GetMeQuery>(GET_ME);
    const [updateUser] = useMutation<UpdateUserMutation, UpdateUserMutationVariables>(UPDATE_USER, {
        onCompleted: () => {
            toast.success("Position updated!");
            setOpen(false);
        },
        onError: (error) => {
            toast.error(error.message);
        },
        refetchQueries: [{ query: GET_ME }],
    });

    // Load preferences on mount
    useEffect(() => {
        // Load radius from cookie
        const cookies = document.cookie.split(';');
        const radiusCookie = cookies.find(c => c.trim().startsWith('matcher_radius='));
        if (radiusCookie) {
            const val = parseInt(radiusCookie.split('=')[1]);
            if (!isNaN(val)) setRadius(val);
        }

        if (meData?.me) {
            if (meData.me.latitude) setLat(meData.me.latitude);
            if (meData.me.longitude) setLng(meData.me.longitude);
        }
    }, [meData]);

    // Save radius to cookie whenever it changes
    useEffect(() => {
        document.cookie = `matcher_radius=${radius}; path=/; max-age=${60 * 60 * 24 * 365}`; // 1 year
    }, [radius]);

    const handleGetLocation = () => {
        setIsLocating(true);
        if (!navigator.geolocation) {
            toast.error("Geolocation is not supported by your browser");
            setIsLocating(false);
            return;
        }

        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const newLat = position.coords.latitude;
                const newLng = position.coords.longitude;
                setLat(newLat);
                setLng(newLng);
                setIsLocating(false);

                // Automatically update db
                if (meData?.me?.id) {
                    await updateUser({
                        variables: {
                            id: meData.me.id,
                            input: {
                                latitude: newLat,
                                longitude: newLng,
                            },
                        },
                    });
                }
            },
            (error) => {
                toast.error("Unable to retrieve your location: " + error.message);
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
                        {hasLocation ? `${radius} km` : "Set Location"}
                    </span>
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Matching Radius & Location</DialogTitle>
                    <DialogDescription>
                        Set your current location and how far you want to look for matches.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-6 py-4">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <Label htmlFor="radius">Search Radius: {radius} km</Label>
                            <Target className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <Slider
                            id="radius"
                            min={1}
                            max={100}
                            step={1}
                            value={[radius]}
                            onValueChange={(val) => setRadius(val[0])}
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
                                {hasLocation ? "Update Position" : "Get Current Position"}
                            </Button>
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}
