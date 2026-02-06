"use client";

import { useSearchParams } from "next/navigation";
import { useState, useEffect, Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SCOPES, scopesRequireProfile } from "@/lib/oauth/config";
import { Questionnaire } from "@/components/questionnaire";
import { CheckCircle2Icon, Loader2Icon, ShieldCheckIcon } from "lucide-react";

interface ClientInfo {
  name: string;
  description?: string;
}

interface UserInfo {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
}

type AuthStep = "auth" | "questionnaire" | "consent";

function AuthorizeContent() {
  const searchParams = useSearchParams();

  const [error, setError] = useState<string | null>(null);
  const [client, setClient] = useState<ClientInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [authLoading, setAuthLoading] = useState(false);

  // Auth state
  const [step, setStep] = useState<AuthStep>("auth");
  const [user, setUser] = useState<UserInfo | null>(null);

  // Login form
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // Signup form
  const [signupData, setSignupData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    birthDate: "",
    gender: "" as "" | "man" | "woman" | "non_binary",
  });

  // OAuth params
  const responseType = searchParams.get("response_type");
  const clientId = searchParams.get("client_id");
  const redirectUri = searchParams.get("redirect_uri");
  const scope = searchParams.get("scope") || "";
  const state = searchParams.get("state") || "";
  const codeChallenge = searchParams.get("code_challenge");
  const codeChallengeMethod = searchParams.get("code_challenge_method") || "S256";

  // Check if scopes require a completed profile
  const requiresProfile = scopesRequireProfile(scope);

  // Check if already authenticated on mount
  useEffect(() => {
    async function checkAuth() {
      try {
        console.log("Checking auth status...");
        const res = await fetch("/api/auth/profile-status", { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          console.log("Auth status:", data);
          if (data.authenticated) {
            setUser(data.user);
            // Only require questionnaire if scopes need it AND user hasn't completed it
            if (requiresProfile && !data.hasProfile) {
              setStep("questionnaire");
            } else {
              setStep("consent");
            }
          }
        }
      } catch {
        // Not authenticated, continue with login
      }
    }
    checkAuth();
  }, [requiresProfile]);

  useEffect(() => {
    async function validateRequest() {
      try {
        // Validate required params
        if (responseType !== "code") {
          setError("Invalid response_type. Only 'code' is supported.");
          setLoading(false);
          return;
        }

        if (!clientId || !redirectUri) {
          setError("Missing required parameters: client_id and redirect_uri");
          setLoading(false);
          return;
        }

        // Fetch client info
        const res = await fetch(`/api/oauth/client-info?client_id=${clientId}`);
        if (!res.ok) {
          setError("Invalid client_id");
          setLoading(false);
          return;
        }

        const clientData = await res.json();
        setClient(clientData);
        setLoading(false);
      } catch {
        setError("Failed to validate request");
        setLoading(false);
      }
    }

    validateRequest();
  }, [responseType, clientId, redirectUri]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: loginEmail, password: loginPassword }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Login failed");
      }

      const data = await res.json();
      setUser(data.user);

      // Check if user has profile
      const statusRes = await fetch("/api/auth/profile-status");
      const status = await statusRes.json();
      // Questionnaire only if scopes require it AND user hasn't completed it
      if (requiresProfile && !status.hasProfile) {
        setStep("questionnaire");
      } else {
        setStep("consent");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setError(null);

    if (signupData.password !== signupData.confirmPassword) {
      setError("Le password non corrispondono");
      setAuthLoading(false);
      return;
    }

    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: signupData.email,
          password: signupData.password,
          firstName: signupData.firstName,
          lastName: signupData.lastName,
          birthDate: signupData.birthDate,
          gender: signupData.gender || undefined,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Signup failed");
      }

      const data = await res.json();
      setUser(data.user);
      // New users need questionnaire only if scopes require it
      setStep(requiresProfile ? "questionnaire" : "consent");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Signup failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleQuestionnaireComplete = async (answers: Record<string, number | string>) => {
    setAuthLoading(true);
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

      setStep("consent");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save assessment");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleAuthorize = async () => {
    setAuthLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/oauth/authorize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_id: clientId,
          redirect_uri: redirectUri,
          scope,
          state,
          code_challenge: codeChallenge,
          code_challenge_method: codeChallengeMethod,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error_description || "Authorization failed");
      }

      const { redirect_url } = await res.json();
      window.location.href = redirect_url;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authorization failed");
      setAuthLoading(false);
    }
  };

  const handleDeny = () => {
    const errorUrl = new URL(redirectUri!);
    errorUrl.searchParams.set("error", "access_denied");
    errorUrl.searchParams.set("error_description", "User denied the request");
    if (state) errorUrl.searchParams.set("state", state);
    window.location.href = errorUrl.toString();
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2Icon className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error && !client) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle className="text-destructive">Errore</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const requestedScopes = scope.split(" ").filter(Boolean);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary mb-4">
            <span className="text-2xl font-bold text-primary-foreground">M</span>
          </div>
          <h1 className="text-xl font-semibold text-foreground">
            Accedi a <span className="text-primary">{client?.name}</span>
          </h1>
          {client?.description && (
            <p className="text-sm text-muted-foreground mt-1">{client.description}</p>
          )}
        </div>

        {/* Step indicator */}
        <div className="flex items-center justify-center gap-2 mb-6">
          {["auth", "questionnaire", "consent"].map((s, i) => (
            <div key={s} className="flex items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${step === s
                  ? "bg-primary text-primary-foreground"
                  : i < ["auth", "questionnaire", "consent"].indexOf(step)
                    ? "bg-primary/30 text-primary"
                    : "bg-muted text-muted-foreground"
                  }`}
              >
                {i < ["auth", "questionnaire", "consent"].indexOf(step) ? (
                  <CheckCircle2Icon className="w-4 h-4" />
                ) : (
                  i + 1
                )}
              </div>
              {i < 2 && (
                <div
                  className={`w-12 h-0.5 mx-1 ${i < ["auth", "questionnaire", "consent"].indexOf(step)
                    ? "bg-primary/50"
                    : "bg-muted"
                    }`}
                />
              )}
            </div>
          ))}
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded-lg mb-4 text-sm">
            {error}
          </div>
        )}

        {/* Step: Auth (Login/Signup) */}
        {step === "auth" && (
          <Card>
            <CardContent className="pt-6">
              <Tabs defaultValue="login" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="login">Accedi</TabsTrigger>
                  <TabsTrigger value="signup">Registrati</TabsTrigger>
                </TabsList>

                <TabsContent value="login" className="mt-6">
                  <form onSubmit={handleLogin} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="login-email">Email</Label>
                      <Input
                        id="login-email"
                        type="email"
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        required
                        placeholder="tu@esempio.com"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="login-password">Password</Label>
                      <Input
                        id="login-password"
                        type="password"
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        required
                        placeholder="••••••••"
                      />
                    </div>
                    <Button type="submit" disabled={authLoading} className="w-full">
                      {authLoading && <Loader2Icon className="w-4 h-4 animate-spin mr-2" />}
                      Accedi
                    </Button>
                  </form>
                </TabsContent>

                <TabsContent value="signup" className="mt-6">
                  <form onSubmit={handleSignup} className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="signup-firstname">Nome</Label>
                        <Input
                          id="signup-firstname"
                          value={signupData.firstName}
                          onChange={(e) =>
                            setSignupData({ ...signupData, firstName: e.target.value })
                          }
                          required
                          placeholder="Mario"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="signup-lastname">Cognome</Label>
                        <Input
                          id="signup-lastname"
                          value={signupData.lastName}
                          onChange={(e) =>
                            setSignupData({ ...signupData, lastName: e.target.value })
                          }
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
                        onChange={(e) =>
                          setSignupData({ ...signupData, email: e.target.value })
                        }
                        required
                        placeholder="tu@esempio.com"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="signup-birthdate">Data di nascita</Label>
                        <Input
                          id="signup-birthdate"
                          type="date"
                          value={signupData.birthDate}
                          onChange={(e) =>
                            setSignupData({ ...signupData, birthDate: e.target.value })
                          }
                          required
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Genere</Label>
                        <Select
                          value={signupData.gender}
                          onValueChange={(v) =>
                            setSignupData({
                              ...signupData,
                              gender: v as "man" | "woman" | "non_binary",
                            })
                          }
                        >
                          <SelectTrigger>
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

                    <div className="space-y-2">
                      <Label htmlFor="signup-password">Password</Label>
                      <Input
                        id="signup-password"
                        type="password"
                        value={signupData.password}
                        onChange={(e) =>
                          setSignupData({ ...signupData, password: e.target.value })
                        }
                        required
                        minLength={8}
                        placeholder="Minimo 8 caratteri"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="signup-confirm">Conferma password</Label>
                      <Input
                        id="signup-confirm"
                        type="password"
                        value={signupData.confirmPassword}
                        onChange={(e) =>
                          setSignupData({ ...signupData, confirmPassword: e.target.value })
                        }
                        required
                        placeholder="Ripeti la password"
                      />
                    </div>

                    <Button type="submit" disabled={authLoading} className="w-full">
                      {authLoading && <Loader2Icon className="w-4 h-4 animate-spin mr-2" />}
                      Crea account
                    </Button>
                  </form>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}

        {/* Step: Questionnaire */}
        {step === "questionnaire" && (
          <div>
            <Card className="mb-4">
              <CardHeader>
                <CardTitle>Completa il tuo profilo</CardTitle>
                <CardDescription>
                  Rispondi a qualche domanda per aiutarci a trovare persone compatibili con te.
                </CardDescription>
              </CardHeader>
            </Card>

            {authLoading ? (
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
                onSkip={() => setStep("consent")}
              />
            )}
          </div>
        )}

        {/* Step: Consent */}
        {step === "consent" && (
          <Card>
            <CardHeader className="text-center">
              <div className="mx-auto mb-2">
                <ShieldCheckIcon className="w-12 h-12 text-primary" />
              </div>
              <CardTitle>Bentornato{user?.firstName ? `, ${user.firstName}` : ""}!</CardTitle>
              <CardDescription>Autorizza l&apos;accesso per continuare</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <p className="text-sm font-medium text-foreground mb-3">
                  {client?.name} avrà accesso a:
                </p>
                <ul className="space-y-2">
                  {requestedScopes.map((s) => (
                    <li key={s} className="flex items-center text-muted-foreground text-sm">
                      <CheckCircle2Icon className="w-4 h-4 text-primary mr-2 flex-shrink-0" />
                      {SCOPES[s as keyof typeof SCOPES] || s}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="flex gap-3">
                <Button onClick={handleDeny} variant="outline" className="flex-1">
                  Rifiuta
                </Button>
                <Button onClick={handleAuthorize} disabled={authLoading} className="flex-1">
                  {authLoading && <Loader2Icon className="w-4 h-4 animate-spin mr-2" />}
                  Autorizza
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        <p className="text-xs text-muted-foreground text-center mt-4">
          Verrai reindirizzato a {redirectUri?.split("?")[0]}
        </p>
      </div>
    </div>
  );
}

export default function AuthorizePage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-background">
          <Loader2Icon className="h-8 w-8 animate-spin text-primary" />
        </div>
      }
    >
      <AuthorizeContent />
    </Suspense>
  );
}
