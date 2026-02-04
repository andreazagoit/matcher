"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { LogIn } from "lucide-react";
import Link from "next/link";
import { signIn } from "next-auth/react";

export default function LoginPage() {
    const handleLogin = () => {
        // Redirect to our OAuth provider
        signIn("matcher", { callbackUrl: "/dashboard" });
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-background p-4">
            <Card className="w-full max-w-md">
                <CardHeader className="space-y-1">
                    <CardTitle className="text-2xl font-bold">Accedi</CardTitle>
                    <CardDescription>
                        Accedi con il tuo account Matcher.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <Button
                        className="w-full"
                        size="lg"
                        onClick={handleLogin}
                    >
                        <LogIn className="w-4 h-4 mr-2" />
                        Accedi con Matcher
                    </Button>
                </CardContent>
                <CardFooter className="flex flex-col gap-4">
                    <div className="text-center text-sm text-muted-foreground">
                        Non hai un account?{" "}
                        <Link href="/signup" className="text-primary hover:underline">
                            Registrati
                        </Link>
                    </div>
                </CardFooter>
            </Card>
        </div>
    );
}
