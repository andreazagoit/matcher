"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Lock, CreditCard } from "lucide-react"

interface PaymentRequiredViewProps {
    spaceName: string
    tierName?: string
    price?: number
    interval?: string
    onPaymentComplete: () => void
}

export function PaymentRequiredView({ spaceName, tierName, price, interval, onPaymentComplete: _ }: PaymentRequiredViewProps) {
    void _;
    const handlePay = async () => {
        // Mock payment flow
        // In real app, this would redirect to Stripe Checkout
        if (!confirm("Proceed to payment?")) return;

        try {
            // Simulate payment success by updating member status to 'active'
            // We need a mutation for this, or just re-join?
            // Usually valid payment webhook updates status. 
            // For now, we might need a debug mutation or manual admin approval simulation.

            // For this MVP, let's assume clicking "Pay" triggers an "Auto-Approve" or similar
            // But we don't have that endpoint. 
            // We implemented `approveMember` for admins.
            // Maybe we can have a `simulatePayment` mutation for dev/demo?

            alert("Payment simulation: Please contact admin to approve your request, or wait for webhook.");
            // onPaymentComplete(); // Actually we can't complete it without backend change.
        } catch (error) {
            console.error(error);
        }
    }

    return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] p-4 text-center">
            <div className="bg-primary/10 p-6 rounded-full mb-6">
                <Lock className="h-12 w-12 text-primary" />
            </div>
            <h2 className="text-3xl font-bold tracking-tight mb-2">Membership Pending</h2>
            <p className="text-muted-foreground max-w-md mb-8">
                You have requested to join <strong>{spaceName}</strong> on the <strong>{tierName || "Premium"}</strong> plan.
                Payment is required to activate your membership.
            </p>

            <Card className="w-full max-w-sm">
                <CardHeader>
                    <CardTitle>Subscription Summary</CardTitle>
                    <CardDescription>Review your plan details</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex justify-between items-center py-2 border-b">
                        <span className="font-medium">{tierName} Plan</span>
                        <span className="font-bold">
                            {price ? (price / 100).toLocaleString("en-EU", { style: "currency", currency: "EUR" }) : "TBD"}
                            /{interval === "one_time" ? "lifetime" : interval}
                        </span>
                    </div>
                    <div className="flex justify-between items-center text-sm text-muted-foreground">
                        <span>Due today</span>
                        <span>{price ? (price / 100).toLocaleString("en-EU", { style: "currency", currency: "EUR" }) : "TBD"}</span>
                    </div>
                </CardContent>
                <CardFooter>
                    <Button className="w-full" onClick={handlePay}>
                        <CreditCard className="mr-2 h-4 w-4" />
                        Proceed to Payment
                    </Button>
                </CardFooter>
            </Card>

            <p className="text-xs text-muted-foreground mt-4">
                Secure payment processing via Stripe.
            </p>
        </div>
    )
}
