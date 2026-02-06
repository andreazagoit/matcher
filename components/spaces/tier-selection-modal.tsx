"use client"

import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"


interface Tier {
    id: string
    name: string
    description?: string
    price: number
    interval: string
}

interface TierSelectionModalProps {
    isOpen: boolean
    onClose: () => void
    tiers: Tier[]
    onSelect: (tierId: string) => void
    isJoining: boolean
}

export function TierSelectionModal({
    isOpen,
    onClose,
    tiers,
    onSelect,
    isJoining
}: TierSelectionModalProps) {
    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="sm:max-w-[800px]">
                <DialogHeader>
                    <DialogTitle>Choose your membership</DialogTitle>
                    <DialogDescription>
                        Select a plan to join this space.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 py-4">
                    {tiers.map((tier) => (
                        <Card key={tier.id} className="flex flex-col relative overflow-hidden">
                            {/* Highlight popular or recommended if needed */}
                            <CardHeader>
                                <CardTitle className="flex justify-between items-center">
                                    <span>{tier.name}</span>
                                </CardTitle>
                                <div className="mt-2">
                                    <span className="text-3xl font-bold">
                                        {(tier.price / 100).toLocaleString("en-EU", { style: "currency", currency: "EUR" })}
                                    </span>
                                    <span className="text-muted-foreground text-sm">
                                        /{tier.interval === "one_time" ? "lifetime" : tier.interval}
                                    </span>
                                </div>
                            </CardHeader>
                            <CardContent className="flex-1">
                                <p className="text-sm text-muted-foreground mb-4">{tier.description}</p>
                                {/* Feature list could go here if structured */}
                            </CardContent>
                            <CardFooter>
                                <Button
                                    className="w-full"
                                    onClick={() => onSelect(tier.id)}
                                    disabled={isJoining}
                                >
                                    {isJoining ? "Joining..." : "Select Plan"}
                                </Button>
                            </CardFooter>
                        </Card>
                    ))}
                </div>
            </DialogContent>
        </Dialog>
    )
}
