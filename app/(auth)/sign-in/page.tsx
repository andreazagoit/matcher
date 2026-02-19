"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authClient } from "@/lib/auth-client";
import {
  ArrowLeftIcon,
  Loader2Icon,
  LogInIcon,
  MailIcon,
} from "lucide-react";
import Link from "next/link";

type LoginStep = "email" | "otp";

export default function SignInPage() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [email, setEmail] = useState("");
  const [step, setStep] = useState<LoginStep>("email");
  const [otp, setOtp] = useState("");

  const fail = (message: string) => {
    setError(message);
    setLoading(false);
  };

  const handleSendOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setError(null);

    try {
      const result = await authClient.emailOtp.sendVerificationOtp({
        email,
        type: "sign-in",
      });

      if (result?.error) {
        fail(
          (result.error as { message?: string }).message || "Invio OTP fallito",
        );
        return;
      }

      setStep("otp");
      setLoading(false);
    } catch (err) {
      fail(err instanceof Error ? err.message : "Invio OTP fallito");
    }
  };

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setError(null);

    try {
      const result = await authClient.signIn.emailOtp({
        email,
        otp,
      });

      if (result?.error) {
        fail(
          (result.error as { message?: string }).message ||
            "Verifica OTP fallita",
        );
        return;
      }

      window.location.href = "/spaces";
    } catch (err) {
      fail(err instanceof Error ? err.message : "Verifica OTP fallita");
    }
  };

  const handleBackToEmail = () => {
    setStep("email");
    setOtp("");
    setError(null);
  };

  return (
    <div className="flex-1 flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary mb-4">
            <span className="text-xl font-bold text-primary-foreground">M</span>
          </div>
          <h1 className="text-xl font-semibold text-foreground">Matcher</h1>
          <p className="text-sm text-muted-foreground mt-1.5">
            Accedi al tuo account
          </p>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 rounded-xl mb-5 text-sm">
            {error}
          </div>
        )}

        <Card className="border-border/50 bg-card/60 backdrop-blur-sm rounded-2xl overflow-hidden">
          <CardHeader>
            <CardTitle>
              {step === "email" ? "Accedi" : "Inserisci il codice"}
            </CardTitle>
            <CardDescription>
              {step === "email"
                ? "Inserisci la tua email per ricevere un codice di accesso"
                : `Abbiamo inviato un codice a ${email}`}
            </CardDescription>
          </CardHeader>

          <CardContent>
            {step === "email" ? (
              <form onSubmit={handleSendOtp} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="login-email">Email</Label>
                  <Input
                    id="login-email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    placeholder="tu@esempio.com"
                    autoComplete="email"
                  />
                </div>

                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <Loader2Icon className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <MailIcon className="w-4 h-4 mr-2" />
                  )}
                  Invia codice
                </Button>
              </form>
            ) : (
              <form onSubmit={handleVerifyOtp} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="login-otp">Codice OTP</Label>
                  <Input
                    id="login-otp"
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={6}
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))}
                    required
                    placeholder="000000"
                    autoComplete="one-time-code"
                    autoFocus
                    className="text-center text-2xl tracking-[0.5em] font-mono"
                  />
                  <p className="text-xs text-muted-foreground">
                    Controlla la tua casella email. Il codice scade tra 5 minuti.
                  </p>
                </div>

                <Button
                  type="submit"
                  disabled={loading || otp.length < 6}
                  className="w-full"
                >
                  {loading ? (
                    <Loader2Icon className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <LogInIcon className="w-4 h-4 mr-2" />
                  )}
                  Accedi
                </Button>

                <Button
                  type="button"
                  variant="ghost"
                  onClick={handleBackToEmail}
                  className="w-full gap-2 text-muted-foreground"
                >
                  <ArrowLeftIcon className="w-4 h-4" />
                  Cambia email
                </Button>
              </form>
            )}

            <div className="mt-6 pt-6 border-t text-center">
              <p className="text-sm text-muted-foreground">
                Non hai un account?{" "}
                <Link
                  href="/sign-up"
                  className="text-primary hover:underline font-medium"
                >
                  Registrati
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
