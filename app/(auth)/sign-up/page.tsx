"use client";

import { useState, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
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
import { signUpSchema } from "@/lib/models/users/validator";
import { ArrowLeftIcon, Loader2Icon, LogInIcon, UserPlusIcon } from "lucide-react";
import Link from "next/link";
import { useLazyQuery } from "@apollo/client/react";
import { CHECK_USERNAME } from "@/lib/models/users/gql";
type FieldErrors = Partial<Record<"username" | "givenName" | "familyName" | "email" | "birthdate" | "gender", string>>;

interface SignupData {
  username: string;
  givenName: string;
  familyName: string;
  email: string;
  birthdate: string;
  gender: "" | "man" | "woman" | "non_binary";
}

const EMPTY_SIGNUP: SignupData = {
  username: "",
  givenName: "",
  familyName: "",
  email: "",
  birthdate: "",
  gender: "",
};

export default function SignUpPageWrapper() {
  return (
    <Suspense>
      <SignUpPage />
    </Suspense>
  );
}

function SignUpPage() {
  const searchParams = useSearchParams();
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Pre-fill email from query param (e.g. redirected from sign-in)
  const [signupData, setSignupData] = useState<SignupData>(() => ({
    ...EMPTY_SIGNUP,
    email: searchParams.get("email") ?? "",
  }));
  const [step, setStep] = useState<"form" | "verify">("form");
  const [otp, setOtp] = useState("");
  const [usernameTaken, setUsernameTaken] = useState(false);
  const usernameDebounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [checkUsername] = useLazyQuery<{ checkUsername: boolean }>(CHECK_USERNAME, {
    fetchPolicy: "network-only",
  });

  const updateSignup = <K extends keyof SignupData>(k: K, v: SignupData[K]) => {
    setSignupData((prev) => ({ ...prev, [k]: v }));
    if (fieldErrors[k]) setFieldErrors((prev) => ({ ...prev, [k]: undefined }));

    if (k === "username") {
      setUsernameTaken(false);
      if (usernameDebounce.current) clearTimeout(usernameDebounce.current);
      const val = v as string;
      if (val.length >= 3) {
        usernameDebounce.current = setTimeout(async () => {
          const { data } = await checkUsername({ variables: { username: val } });
          setUsernameTaken(data?.checkUsername ?? false);
        }, 400);
      }
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setSubmitError(null);
    setFieldErrors({});

    const parsed = signUpSchema.safeParse(signupData);
    if (!parsed.success) {
      const errors: FieldErrors = {};
      for (const issue of parsed.error.issues) {
        const field = issue.path[0] as keyof SignupData;
        if (!errors[field]) errors[field] = issue.message;
      }
      setFieldErrors(errors);
      setLoading(false);
      return;
    }

    try {
      const randomPassword = crypto.randomUUID() + "!Aa1";

      const result = await authClient.signUp.email({
        email: signupData.email,
        password: randomPassword,
        name: `${signupData.givenName} ${signupData.familyName}`,
        username: signupData.username,
        givenName: signupData.givenName,
        familyName: signupData.familyName,
        birthdate: signupData.birthdate,
        gender: signupData.gender,
      } as Parameters<typeof authClient.signUp.email>[0]);

      if (result?.error) {
        setSubmitError(
          (result.error as { message?: string }).message || "Registrazione fallita",
        );
        setLoading(false);
        return;
      }

      // Account created — now verify email via OTP
      setStep("verify");
      setLoading(false);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Registrazione fallita");
      setLoading(false);
    }
  };

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setSubmitError(null);

    try {
      const result = await authClient.emailOtp.verifyEmail({
        email: signupData.email,
        otp,
      });

      if (result?.error) {
        setSubmitError(
          (result.error as { message?: string }).message || "Codice non valido",
        );
        setLoading(false);
        return;
      }

      window.location.href = `/users/${signupData.username}`;
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Verifica fallita");
      setLoading(false);
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

        {submitError && (
          <div className="bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 rounded-xl mb-5 text-sm">
            {submitError}
          </div>
        )}

        <Card className="border-border/50 bg-card/60 backdrop-blur-sm rounded-2xl overflow-hidden">
          <CardHeader>
            <CardTitle>{step === "form" ? "Registrati" : "Verifica email"}</CardTitle>
            <CardDescription>
              {step === "form"
                ? "Compila i dati per creare il tuo account"
                : `Abbiamo inviato un codice a ${signupData.email}. Inseriscilo per attivare l'account.`}
            </CardDescription>
          </CardHeader>

          <CardContent>
            {step === "verify" ? (
              <form onSubmit={handleVerifyOtp} className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium" htmlFor="verify-otp">Codice OTP</label>
                  <input
                    id="verify-otp"
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
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-center text-2xl tracking-[0.5em] font-mono ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  />
                  <p className="text-xs text-muted-foreground">
                    Il codice scade tra 5 minuti.
                  </p>
                </div>

                <Button type="submit" disabled={loading || otp.length < 6} className="w-full">
                  {loading ? (
                    <Loader2Icon className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <LogInIcon className="w-4 h-4 mr-2" />
                  )}
                  Verifica e accedi
                </Button>
              </form>
            ) : (
            <>
            <form onSubmit={handleSignup} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="signup-username">Username</Label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">@</span>
                  <Input
                    id="signup-username"
                    value={signupData.username}
                    onChange={(e) => updateSignup("username", e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ""))}
                    placeholder="mario_rossi"
                    className={`pl-7 ${fieldErrors.username || usernameTaken ? "border-destructive" : ""}`}
                    maxLength={30}
                    autoComplete="username"
                  />
                </div>
                {fieldErrors.username ? (
                  <p className="text-xs text-destructive">{fieldErrors.username}</p>
                ) : usernameTaken ? (
                  <p className="text-xs text-destructive">Username già in uso</p>
                ) : (
                  <p className="text-xs text-muted-foreground">3–30 caratteri, solo lettere minuscole, numeri e _</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="signup-givenname">Nome</Label>
                  <Input
                    id="signup-givenname"
                    value={signupData.givenName}
                    onChange={(e) => updateSignup("givenName", e.target.value)}
                    placeholder="Mario"
                    className={fieldErrors.givenName ? "border-destructive" : ""}
                  />
                  {fieldErrors.givenName && <p className="text-xs text-destructive">{fieldErrors.givenName}</p>}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="signup-familyname">Cognome</Label>
                  <Input
                    id="signup-familyname"
                    value={signupData.familyName}
                    onChange={(e) => updateSignup("familyName", e.target.value)}
                    placeholder="Rossi"
                    className={fieldErrors.familyName ? "border-destructive" : ""}
                  />
                  {fieldErrors.familyName && <p className="text-xs text-destructive">{fieldErrors.familyName}</p>}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="signup-email">Email</Label>
                <Input
                  id="signup-email"
                  type="email"
                  value={signupData.email}
                  onChange={(e) => updateSignup("email", e.target.value)}
                  placeholder="tu@esempio.com"
                  autoComplete="email"
                  className={fieldErrors.email ? "border-destructive" : ""}
                />
                {fieldErrors.email && <p className="text-xs text-destructive">{fieldErrors.email}</p>}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="signup-birthdate">Data di nascita</Label>
                  <Input
                    id="signup-birthdate"
                    type="date"
                    value={signupData.birthdate}
                    onChange={(e) => updateSignup("birthdate", e.target.value)}
                    className={`dark:[color-scheme:dark] ${fieldErrors.birthdate ? "border-destructive" : ""}`}
                  />
                  {fieldErrors.birthdate && <p className="text-xs text-destructive">{fieldErrors.birthdate}</p>}
                </div>
                <div className="space-y-2">
                  <Label>Genere</Label>
                  <Select
                    value={signupData.gender}
                    onValueChange={(v) => updateSignup("gender", v as SignupData["gender"])}
                  >
                    <SelectTrigger className={`w-full ${fieldErrors.gender ? "border-destructive" : ""}`}>
                      <SelectValue placeholder="Seleziona..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="man">Uomo</SelectItem>
                      <SelectItem value="woman">Donna</SelectItem>
                      <SelectItem value="non_binary">Non binario</SelectItem>
                    </SelectContent>
                  </Select>
                  {fieldErrors.gender && <p className="text-xs text-destructive">{fieldErrors.gender}</p>}
                </div>
              </div>

              <Button type="submit" disabled={loading || usernameTaken} className="w-full">
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
            </>
            )}
          </CardContent>
        </Card>

        <div className="mt-6 text-center">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground">
              <ArrowLeftIcon className="w-4 h-4" />
              Indietro
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
