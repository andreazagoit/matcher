"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Questionnaire } from "@/components/questionnaire";
import { Loader2Icon } from "lucide-react";
import { useSession } from "next-auth/react";

interface ProfileCompletionProps {
    onComplete?: () => void;
    title?: string;
    description?: string;
    className?: string;
}

export function ProfileCompletion({
    onComplete,
    title = "Completa il tuo profilo",
    description,
    className
}: ProfileCompletionProps) {
    const { data: session } = useSession();
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const defaultDescription = `Ciao ${session?.user?.name || ""}. Rispondi a qualche domanda per aiutarci a trovare persone compatibili con te.`;

    const handleQuestionnaireComplete = async (answers: Record<string, number | string>) => {
        setLoading(true);
        setError(null);

        try {
            const res = await fetch("/api/auth/complete-assessment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ answers }),
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || "Failed to save assessment");
            }

            // If an onComplete callback is provided, use it.
            if (onComplete) {
                onComplete();
                return;
            }

            // Default behavior: Check if there's an OAuth redirect pending
            const oauthRedirect = sessionStorage.getItem("oauth_redirect");
            if (oauthRedirect) {
                sessionStorage.removeItem("oauth_redirect");
                window.location.href = oauthRedirect;
            } else {
                router.push("/spaces");
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save assessment");
            setLoading(false);
        }
    };

    const handleSkip = () => {
        const oauthRedirect = sessionStorage.getItem("oauth_redirect");
        if (oauthRedirect) {
            sessionStorage.removeItem("oauth_redirect");
            window.location.href = oauthRedirect;
        } else {
            router.push("/spaces");
        }
    };

    return (
        <div className={className}>
            <Card className="mb-6">
                <CardHeader>
                    <CardTitle>{title}</CardTitle>
                    <CardDescription>
                        {description || defaultDescription}
                    </CardDescription>
                </CardHeader>
            </Card>

            {error && (
                <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded-lg mb-4 text-sm">
                    {error}
                </div>
            )}

            {loading ? (
                <Card>
                    <CardContent className="py-12 text-center">
                        <Loader2Icon className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
                        <p className="text-muted-foreground">Stiamo creando il tuo profilo...</p>
                        <p className="text-xs text-muted-foreground mt-1">
                            Questo potrebbe richiedere qualche secondo
                        </p>
                    </CardContent>
                </Card>
            ) : (
                <Questionnaire
                    onComplete={handleQuestionnaireComplete}
                    onSkip={handleSkip}
                />
            )}
        </div>
    );
}
