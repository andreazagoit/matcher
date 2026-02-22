"use client";

import { useState, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTranslations } from "next-intl";
import { authClient } from "@/lib/auth-client";
import { signUpSchema, signupFormSchema, type SignupFormData, SUPPORTED_LANGUAGES } from "@/lib/models/users/validator";
import {
  genderEnum, sexualOrientationEnum, relationshipIntentEnum, relationshipStyleEnum,
  hasChildrenEnum, wantsChildrenEnum, smokingEnum, drinkingEnum, activityLevelEnum,
  religionEnum, educationLevelEnum, ethnicityEnum,
} from "@/lib/models/users/schema";
import { ArrowLeftIcon, ArrowRightIcon, Loader2Icon, UserPlusIcon } from "lucide-react";
import Link from "next/link";
import { useLazyQuery } from "@apollo/client/react";
import { CHECK_USERNAME } from "@/lib/models/users/gql";

type Step = "identity" | "intent" | "about" | "background" | "lifestyle" | "account" | "verify";
const STEPS: Step[] = ["identity", "intent", "about", "background", "lifestyle", "account", "verify"];

type FieldErrors = Partial<Record<"name" | "birthdate" | "gender" | "username" | "email", string>>;

const EMPTY: SignupFormData = {
  name: "", birthdate: "", username: "", email: "",
  sexualOrientation: [], relationshipIntent: [], languages: [],
};

export default function SignUpPage() {
  return <Suspense><SignUpForm /></Suspense>;
}

function StepIndicator({ current }: { current: Step }) {
  const idx = STEPS.indexOf(current);
  return (
    <div className="flex items-center justify-center gap-1.5 mb-6">
      {STEPS.map((s, i) => (
        <div key={s} className="flex items-center gap-1.5">
          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-semibold transition-colors ${
            i < idx ? "bg-primary text-primary-foreground"
            : i === idx ? "bg-primary text-primary-foreground ring-2 ring-primary/30"
            : "bg-muted text-muted-foreground"
          }`}>
            {i < idx ? "✓" : i + 1}
          </div>
          {i < STEPS.length - 1 && (
            <div className={`h-px w-4 transition-colors ${i < idx ? "bg-primary" : "bg-muted"}`} />
          )}
        </div>
      ))}
    </div>
  );
}

function SignUpForm() {
  const tEnums = useTranslations("enums");
  const searchParams = useSearchParams();
  const [data, setData] = useState<SignupFormData>({ ...EMPTY, email: searchParams.get("email") ?? "" });
  const [step, setStep] = useState<Step>("identity");
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [otp, setOtp] = useState("");
  const [usernameTaken, setUsernameTaken] = useState(false);
  const usernameDebounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [checkUsername] = useLazyQuery<{ checkUsername: boolean }>(CHECK_USERNAME, { fetchPolicy: "network-only" });

  const toggle = (k: "sexualOrientation" | "relationshipIntent" | "languages", v: string) => {
    setData((prev) => {
      const arr = prev[k] as string[];
      return { ...prev, [k]: arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v] };
    });
  };

  const set = <K extends keyof SignupFormData>(k: K, v: SignupFormData[K]) => {
    setData((prev) => ({ ...prev, [k]: v }));
    if (k in fieldErrors) setFieldErrors((prev) => ({ ...prev, [k]: undefined }));
    if (k === "username") {
      setUsernameTaken(false);
      if (usernameDebounce.current) clearTimeout(usernameDebounce.current);
      const val = v as string;
      if (val.length >= 3) {
        usernameDebounce.current = setTimeout(async () => {
          const { data: d } = await checkUsername({ variables: { username: val } });
          setUsernameTaken(d?.checkUsername ?? false);
        }, 400);
      }
    }
  };

  const next = () => setStep(STEPS[STEPS.indexOf(step) + 1]);
  const back = () => { setStep(STEPS[STEPS.indexOf(step) - 1]); setSubmitError(null); };

  const handleIdentityNext = (e: React.FormEvent) => {
    e.preventDefault();
    const partial = signUpSchema.pick({ name: true, birthdate: true, gender: true });
    const parsed = partial.safeParse({ name: data.name, birthdate: data.birthdate, gender: data.gender });
    if (!parsed.success) {
      const errors: FieldErrors = {};
      for (const issue of parsed.error.issues) errors[issue.path[0] as keyof FieldErrors] ??= issue.message;
      setFieldErrors(errors);
      return;
    }
    setFieldErrors({});
    next();
  };

  // Step 5: crea account → better-auth invia OTP automaticamente
  const handleAccountNext = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading || usernameTaken) return;
    setSubmitError(null);
    const partial = signUpSchema.pick({ email: true, username: true });
    const parsed = partial.safeParse({ email: data.email, username: data.username });
    if (!parsed.success) {
      const errors: FieldErrors = {};
      for (const issue of parsed.error.issues) errors[issue.path[0] as keyof FieldErrors] ??= issue.message;
      setFieldErrors(errors);
      return;
    }
    setFieldErrors({});
    setLoading(true);
    try {
      const randomPassword = crypto.randomUUID() + "!Aa1";
      const result = await authClient.signUp.email({
        email: data.email,
        password: randomPassword,
        name: data.name,
        username: data.username,
        birthdate: data.birthdate,
        gender: data.gender,
        ...(data.sexualOrientation.length && { sexualOrientation: data.sexualOrientation }),
        ...(data.relationshipIntent.length && { relationshipIntent: data.relationshipIntent }),
        ...(data.relationshipStyle && { relationshipStyle: data.relationshipStyle }),
        ...(data.hasChildren && { hasChildren: data.hasChildren }),
        ...(data.wantsChildren && { wantsChildren: data.wantsChildren }),
        ...(data.smoking && { smoking: data.smoking }),
        ...(data.drinking && { drinking: data.drinking }),
        ...(data.activityLevel && { activityLevel: data.activityLevel }),
        ...(data.religion && { religion: data.religion }),
        ...(data.heightCm && { heightCm: parseInt(data.heightCm) }),
        ...(data.jobTitle && { jobTitle: data.jobTitle }),
        ...(data.educationLevel && { educationLevel: data.educationLevel }),
        ...(data.ethnicity && { ethnicity: data.ethnicity }),
        ...(data.languages.length && { languages: data.languages }),
      } as Parameters<typeof authClient.signUp.email>[0]);

      if (result?.error) {
        const message = (result.error as { message?: string }).message ?? "";
        if (message.toLowerCase().includes("username")) setFieldErrors({ username: message });
        else if (message.toLowerCase().includes("email")) setFieldErrors({ email: message });
        else setSubmitError(message || "Registrazione fallita");
        setLoading(false);
        return;
      }
      // Account creato — better-auth ha già inviato l'OTP di verifica
      next();
      setLoading(false);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Registrazione fallita");
      setLoading(false);
    }
  };

  // Step 6: verifica OTP
  const handleVerifyAndCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setSubmitError(null);
    try {
      const result = await authClient.emailOtp.verifyEmail({ email: data.email, otp });
      if (result?.error) {
        setSubmitError((result.error as { message?: string }).message || "Codice non valido");
        setLoading(false);
        return;
      }
      window.location.href = `/users/${data.username}`;
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Verifica fallita");
      setLoading(false);
    }
  };

  const BackButton = () => (
    <Button type="button" variant="outline" onClick={back} className="gap-2">
      <ArrowLeftIcon className="w-4 h-4" />
    </Button>
  );

  return (
    <div className="flex-1 flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary mb-4">
            <span className="text-xl font-bold text-primary-foreground">M</span>
          </div>
          <h1 className="text-xl font-semibold">Matcher</h1>
          <p className="text-sm text-muted-foreground mt-1.5">Crea un nuovo account</p>
        </div>

        <StepIndicator current={step} />

        {submitError && (
          <div className="bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 rounded-xl mb-5 text-sm">
            {submitError}
          </div>
        )}

        <Card className="border-border/50 bg-card/60 backdrop-blur-sm rounded-2xl overflow-hidden">
          <CardHeader>
            {step === "identity" && <><CardTitle>Chi sei</CardTitle><CardDescription>Qualche informazione di base su di te</CardDescription></>}
            {step === "intent" && <><CardTitle>Cosa cerchi</CardTitle><CardDescription>Puoi cambiarlo in qualsiasi momento</CardDescription></>}
            {step === "about" && <><CardTitle>Su di te</CardTitle><CardDescription>Tutti i campi sono opzionali</CardDescription></>}
            {step === "background" && <><CardTitle>Background</CardTitle><CardDescription>Tutti i campi sono opzionali</CardDescription></>}
            {step === "lifestyle" && <><CardTitle>Stile di vita</CardTitle><CardDescription>Tutti i campi sono opzionali</CardDescription></>}
            {step === "account" && <><CardTitle>Crea il tuo account</CardTitle><CardDescription>Ti invieremo un codice di verifica all&apos;email</CardDescription></>}
            {step === "verify" && <><CardTitle>Verifica email</CardTitle><CardDescription>Abbiamo inviato un codice a {data.email}</CardDescription></>}
          </CardHeader>

          <CardContent>
            {/* ── Step 1: Identity ── */}
            {step === "identity" && (
              <form onSubmit={handleIdentityNext} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Nome</Label>
                  <Input id="name" value={data.name} onChange={(e) => set("name", e.target.value)} placeholder="Mario" autoFocus className={fieldErrors.name ? "border-destructive" : ""} />
                  {fieldErrors.name && <p className="text-xs text-destructive">{fieldErrors.name}</p>}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="birthdate">Data di nascita</Label>
                  <Input id="birthdate" type="date" value={data.birthdate} onChange={(e) => set("birthdate", e.target.value)} className={`dark:[color-scheme:dark] ${fieldErrors.birthdate ? "border-destructive" : ""}`} />
                  {fieldErrors.birthdate && <p className="text-xs text-destructive">{fieldErrors.birthdate}</p>}
                </div>
                <div className="space-y-2">
                  <Label>Genere</Label>
                  <Select value={data.gender ?? ""} onValueChange={(v) => set("gender", v as SignupFormData["gender"])}>
                    <SelectTrigger className={fieldErrors.gender ? "border-destructive" : ""}><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                    <SelectContent>
                      {genderEnum.enumValues.map((v) => (
                        <SelectItem key={v} value={v}>{tEnums(`gender.${v}` as Parameters<typeof tEnums>[0])}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {fieldErrors.gender && <p className="text-xs text-destructive">{fieldErrors.gender}</p>}
                </div>
                <Button type="submit" className="w-full gap-2">Continua <ArrowRightIcon className="w-4 h-4" /></Button>
                <div className="pt-4 border-t text-center text-sm text-muted-foreground">
                  Hai già un account?{" "}
                  <Link href="/sign-in" className="text-primary hover:underline font-medium">Accedi</Link>
                </div>
              </form>
            )}

            {/* ── Step 2: Intent ── */}
            {step === "intent" && (
              <form onSubmit={(e) => { e.preventDefault(); next(); }} className="space-y-4">
                <div className="space-y-2">
                  <Label>Cosa cerchi? <span className="text-muted-foreground font-normal">(puoi scegliere più opzioni)</span></Label>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {relationshipIntentEnum.enumValues.map((v: string) => {
                      const active = (data.relationshipIntent as string[]).includes(v);
                      return (
                        <button key={v} type="button" onClick={() => toggle("relationshipIntent", v)}
                          className={["inline-flex items-center rounded-full border px-4 py-1.5 text-sm font-medium transition-colors",
                            active ? "bg-foreground text-background border-foreground" : "text-muted-foreground hover:border-foreground/40"].join(" ")}>
                          {tEnums(`relationshipIntent.${v}` as Parameters<typeof tEnums>[0])}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Struttura relazionale</Label>
                  <Select value={data.relationshipStyle ?? ""} onValueChange={(v) => set("relationshipStyle", v as SignupFormData["relationshipStyle"])}>
                    <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                    <SelectContent>
                      {relationshipStyleEnum.enumValues.map((v) => (
                        <SelectItem key={v} value={v}>{tEnums(`relationshipStyle.${v}` as Parameters<typeof tEnums>[0])}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex gap-2">
                  <BackButton />
                  <Button type="submit" className="flex-1 gap-2">Continua <ArrowRightIcon className="w-4 h-4" /></Button>
                </div>
              </form>
            )}

            {/* ── Step 3: About ── */}
            {step === "about" && (
              <form onSubmit={(e) => { e.preventDefault(); next(); }} className="space-y-4">
                <div className="space-y-2">
                  <Label>Orientamento sessuale <span className="text-muted-foreground font-normal">(puoi scegliere più opzioni)</span></Label>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {sexualOrientationEnum.enumValues.map((v: string) => {
                      const active = (data.sexualOrientation as string[]).includes(v);
                      return (
                        <button key={v} type="button" onClick={() => toggle("sexualOrientation", v)}
                          className={["inline-flex items-center rounded-full border px-4 py-1.5 text-sm font-medium transition-colors",
                            active ? "bg-foreground text-background border-foreground" : "text-muted-foreground hover:border-foreground/40"].join(" ")}>
                          {tEnums(`sexualOrientation.${v}` as Parameters<typeof tEnums>[0])}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>{tEnums("hasChildrenLabel" as Parameters<typeof tEnums>[0])}</Label>
                    <Select value={data.hasChildren ?? ""} onValueChange={(v) => set("hasChildren", v as SignupFormData["hasChildren"])}>
                      <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                      <SelectContent>
                        {hasChildrenEnum.enumValues.map((v) => (
                          <SelectItem key={v} value={v}>{tEnums(`hasChildren.${v}` as Parameters<typeof tEnums>[0])}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>{tEnums("wantsChildrenLabel" as Parameters<typeof tEnums>[0])}</Label>
                    <Select value={data.wantsChildren ?? ""} onValueChange={(v) => set("wantsChildren", v as SignupFormData["wantsChildren"])}>
                      <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                      <SelectContent>
                        {wantsChildrenEnum.enumValues.map((v) => (
                          <SelectItem key={v} value={v}>{tEnums(`wantsChildren.${v}` as Parameters<typeof tEnums>[0])}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="flex gap-2">
                  <BackButton />
                  <Button type="submit" className="flex-1 gap-2">Continua <ArrowRightIcon className="w-4 h-4" /></Button>
                </div>
              </form>
            )}

            {/* ── Step 4: Background ── */}
            {step === "background" && (
              <form onSubmit={(e) => { e.preventDefault(); next(); }} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="jobTitle">Professione</Label>
                  <Input id="jobTitle" value={data.jobTitle ?? ""} onChange={(e) => set("jobTitle", e.target.value || undefined)} placeholder="es. Designer, Ingegnere…" />
                </div>
                <div className="space-y-2">
                  <Label>{tEnums("educationLevelLabel" as Parameters<typeof tEnums>[0])}</Label>
                  <Select value={data.educationLevel ?? ""} onValueChange={(v) => set("educationLevel", v as SignupFormData["educationLevel"])}>
                    <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                    <SelectContent>
                      {educationLevelEnum.enumValues.map((v) => (
                        <SelectItem key={v} value={v}>{tEnums(`educationLevel.${v}` as Parameters<typeof tEnums>[0])}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>{tEnums("ethnicityLabel" as Parameters<typeof tEnums>[0])}</Label>
                  <Select value={data.ethnicity ?? ""} onValueChange={(v) => set("ethnicity", v as SignupFormData["ethnicity"])}>
                    <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                    <SelectContent>
                      {ethnicityEnum.enumValues.map((v) => (
                        <SelectItem key={v} value={v}>{tEnums(`ethnicity.${v}` as Parameters<typeof tEnums>[0])}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>{tEnums("languagesLabel" as Parameters<typeof tEnums>[0])} <span className="text-muted-foreground font-normal">(puoi scegliere più opzioni)</span></Label>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {SUPPORTED_LANGUAGES.map((v: string) => {
                      const active = (data.languages as string[]).includes(v);
                      return (
                        <button key={v} type="button" onClick={() => toggle("languages", v)}
                          className={["inline-flex items-center rounded-full border px-4 py-1.5 text-sm font-medium transition-colors",
                            active ? "bg-foreground text-background border-foreground" : "text-muted-foreground hover:border-foreground/40"].join(" ")}>
                          {tEnums(`language.${v}` as Parameters<typeof tEnums>[0])}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="flex gap-2">
                  <BackButton />
                  <Button type="submit" className="flex-1 gap-2">Continua <ArrowRightIcon className="w-4 h-4" /></Button>
                </div>
              </form>
            )}

            {/* ── Step 5: Lifestyle ── */}
            {step === "lifestyle" && (
              <form onSubmit={(e) => { e.preventDefault(); next(); }} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Fumo</Label>
                    <Select value={data.smoking ?? ""} onValueChange={(v) => set("smoking", v as SignupFormData["smoking"])}>
                      <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="never">Mai</SelectItem>
                        <SelectItem value="sometimes">A volte</SelectItem>
                        <SelectItem value="regularly">Spesso</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Alcol</Label>
                    <Select value={data.drinking ?? ""} onValueChange={(v) => set("drinking", v as SignupFormData["drinking"])}>
                      <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="never">Mai</SelectItem>
                        <SelectItem value="sometimes">A volte</SelectItem>
                        <SelectItem value="regularly">Spesso</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Attività fisica</Label>
                  <Select value={data.activityLevel ?? ""} onValueChange={(v) => set("activityLevel", v as SignupFormData["activityLevel"])}>
                    <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sedentary">Sedentario</SelectItem>
                      <SelectItem value="light">Leggera</SelectItem>
                      <SelectItem value="moderate">Moderata</SelectItem>
                      <SelectItem value="active">Attivo</SelectItem>
                      <SelectItem value="very_active">Molto attivo</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Religione</Label>
                    <Select value={data.religion ?? ""} onValueChange={(v) => set("religion", v as SignupFormData["religion"])}>
                      <SelectTrigger><SelectValue placeholder="Seleziona..." /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">Nessuna</SelectItem>
                        <SelectItem value="christian">Cristiana</SelectItem>
                        <SelectItem value="muslim">Islamica</SelectItem>
                        <SelectItem value="jewish">Ebraica</SelectItem>
                        <SelectItem value="buddhist">Buddhista</SelectItem>
                        <SelectItem value="hindu">Induista</SelectItem>
                        <SelectItem value="spiritual">Spirituale</SelectItem>
                        <SelectItem value="other">Altra</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="height">Altezza (cm)</Label>
                    <Input id="height" type="number" min={100} max={250} value={data.heightCm ?? ""} onChange={(e) => set("heightCm", e.target.value)} placeholder="175" />
                  </div>
                </div>
                <div className="flex gap-2">
                  <BackButton />
                  <Button type="submit" className="flex-1 gap-2">Continua <ArrowRightIcon className="w-4 h-4" /></Button>
                </div>
              </form>
            )}

            {/* ── Step 5: Account ── */}
            {step === "account" && (
              <form onSubmit={handleAccountNext} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" value={data.email} onChange={(e) => set("email", e.target.value)} placeholder="tu@esempio.com" autoComplete="email" autoFocus className={fieldErrors.email ? "border-destructive" : ""} />
                  {fieldErrors.email && <p className="text-xs text-destructive">{fieldErrors.email}</p>}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">@</span>
                    <Input id="username" value={data.username} onChange={(e) => set("username", e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ""))} placeholder="mario_rossi" className={`pl-7 ${fieldErrors.username || usernameTaken ? "border-destructive" : ""}`} maxLength={30} autoComplete="username" />
                  </div>
                  {fieldErrors.username ? (
                    <p className="text-xs text-destructive">{fieldErrors.username}</p>
                  ) : usernameTaken ? (
                    <p className="text-xs text-destructive">Username già in uso</p>
                  ) : (
                    <p className="text-xs text-muted-foreground">3–30 caratteri, solo lettere minuscole, numeri e _</p>
                  )}
                </div>
                <div className="flex gap-2">
                  <BackButton />
                  <Button type="submit" disabled={loading || usernameTaken} className="flex-1 gap-2">
                    {loading ? <Loader2Icon className="w-4 h-4 animate-spin" /> : <ArrowRightIcon className="w-4 h-4" />}
                    Invia codice
                  </Button>
                </div>
              </form>
            )}

            {/* ── Step 6: Verify ── */}
            {step === "verify" && (
              <form onSubmit={handleVerifyAndCreate} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="otp">Codice OTP</Label>
                  <input
                    id="otp"
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
                  <p className="text-xs text-muted-foreground">Il codice scade tra 5 minuti.</p>
                </div>
                <div className="flex gap-2">
                  <Button type="button" variant="outline" onClick={() => { back(); setOtp(""); }}>
                    <ArrowLeftIcon className="w-4 h-4" />
                  </Button>
                  <Button type="submit" disabled={loading || otp.length < 6} className="flex-1 gap-2">
                    {loading ? <Loader2Icon className="w-4 h-4 animate-spin" /> : <UserPlusIcon className="w-4 h-4" />}
                    Verifica e crea account
                  </Button>
                </div>
              </form>
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
