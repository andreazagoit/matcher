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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { authClient } from "@/lib/auth-client";
import { Loader2Icon, UserPlusIcon } from "lucide-react";
import Link from "next/link";

type Gender = "" | "man" | "woman" | "non_binary";

interface SignupData {
  givenName: string;
  familyName: string;
  email: string;
  birthdate: string;
  gender: Gender;
}

const EMPTY_SIGNUP: SignupData = {
  givenName: "",
  familyName: "",
  email: "",
  birthdate: "",
  gender: "",
};

export default function SignUpPage() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [signupData, setSignupData] = useState<SignupData>(EMPTY_SIGNUP);

  const fail = (message: string) => {
    setError(message);
    setLoading(false);
  };

  const updateSignup = <K extends keyof SignupData>(k: K, v: SignupData[K]) =>
    setSignupData((prev) => ({ ...prev, [k]: v }));

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setError(null);

    if (
      !signupData.birthdate ||
      !/^\d{4}-\d{2}-\d{2}$/.test(signupData.birthdate)
    ) {
      fail("Inserisci una data di nascita valida");
      return;
    }
    if (!signupData.gender) {
      fail("Seleziona il genere");
      return;
    }

    try {
      const randomPassword = crypto.randomUUID() + "!Aa1";

      const result = await authClient.signUp.email({
        email: signupData.email,
        password: randomPassword,
        name: `${signupData.givenName} ${signupData.familyName}`,
        givenName: signupData.givenName,
        familyName: signupData.familyName,
        birthdate: signupData.birthdate,
        gender: signupData.gender,
      } as Parameters<typeof authClient.signUp.email>[0]);

      if (result?.error) {
        fail(
          (result.error as { message?: string }).message ||
            "Registrazione fallita",
        );
        return;
      }

      window.location.href = "/discover";
    } catch (err) {
      fail(err instanceof Error ? err.message : "Registrazione fallita");
    }
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
            Crea un nuovo account
          </p>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 rounded-xl mb-5 text-sm">
            {error}
          </div>
        )}

        <Card className="border-border/50 bg-card/60 backdrop-blur-sm rounded-2xl overflow-hidden">
          <CardHeader>
            <CardTitle>Registrati</CardTitle>
            <CardDescription>
              Compila i dati per creare il tuo account
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSignup} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="signup-givenname">Nome</Label>
                  <Input
                    id="signup-givenname"
                    value={signupData.givenName}
                    onChange={(e) => updateSignup("givenName", e.target.value)}
                    required
                    placeholder="Mario"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="signup-familyname">Cognome</Label>
                  <Input
                    id="signup-familyname"
                    value={signupData.familyName}
                    onChange={(e) => updateSignup("familyName", e.target.value)}
                    required
                    placeholder="Rossi"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="signup-email">Email</Label>
                <Input
                  id="signup-email"
                  type="email"
                  value={signupData.email}
                  onChange={(e) => updateSignup("email", e.target.value)}
                  required
                  placeholder="tu@esempio.com"
                  autoComplete="email"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="signup-birthdate">Data di nascita</Label>
                  <Input
                    id="signup-birthdate"
                    type="date"
                    value={signupData.birthdate}
                    onChange={(e) => updateSignup("birthdate", e.target.value)}
                    required
                    className="dark:[color-scheme:dark]"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Genere</Label>
                  <Select
                    value={signupData.gender}
                    onValueChange={(v) => updateSignup("gender", v as Gender)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Seleziona..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="man">Uomo</SelectItem>
                      <SelectItem value="woman">Donna</SelectItem>
                      <SelectItem value="non_binary">Non binario</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button type="submit" disabled={loading} className="w-full">
                {loading ? (
                  <Loader2Icon className="w-4 h-4 animate-spin mr-2" />
                ) : (
                  <UserPlusIcon className="w-4 h-4 mr-2" />
                )}
                Crea account
              </Button>
            </form>

            <div className="mt-6 pt-6 border-t text-center">
              <p className="text-sm text-muted-foreground">
                Hai già un account?{" "}
                <Link
                  href="/sign-in"
                  className="text-primary hover:underline font-medium"
                >
                  Accedi
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
