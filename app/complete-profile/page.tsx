"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Questionnaire } from "@/components/questionnaire";
import { Loader2Icon } from "lucide-react";

export default function CompleteProfilePage() {
    const { data: session, status } = useSession();
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Redirect if not authenticated
    useEffect(() => {
        if (status === "unauthenticated") {
            router.push("/login");
        }
    }, [status, router]);

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

            // Check if there's an OAuth redirect pending
            const oauthRedirect = sessionStorage.getItem("oauth_redirect");
            if (oauthRedirect) {
                sessionStorage.removeItem("oauth_redirect");
                window.location.href = oauthRedirect;
            } else {
                router.push("/dashboard");
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save assessment");
            setLoading(false);
        }
    };

    if (status === "loading") {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <Loader2Icon className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    if (status === "unauthenticated") {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <Loader2Icon className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background py-8">
            <div className="max-w-2xl mx-auto px-4">
                <Card className="mb-6">
                    <CardHeader>
                        <CardTitle>Completa il tuo profilo</CardTitle>
                        <CardDescription>
                            Ciao {session?.user?.name}! Rispondi a qualche domanda per aiutarci a trovare persone compatibili con te.
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
                        onSkip={() => {
                            const oauthRedirect = sessionStorage.getItem("oauth_redirect");
                            if (oauthRedirect) {
                                sessionStorage.removeItem("oauth_redirect");
                                window.location.href = oauthRedirect;
                            } else {
                                router.push("/dashboard");
                            }
                        }}
                    />
                )}
            </div>
        </div>
    );
}
